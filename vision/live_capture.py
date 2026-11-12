"""Continuous screen capture for the GUI agent.

``LiveCapture`` runs a worker thread that keeps grabbing the screen and holds
the most recent frame, so callers get it instantly with ``latest()`` instead of
paying for a capture on every step. It also measures the achieved capture fps.

The preferred backend is ``dxcam`` (Desktop Duplication API, 60 fps or more);
``mss`` is the fallback when ``dxcam`` is not installed, fails to initialise
or breaks while grabbing. Backends are created, used and closed inside the
worker thread, because both libraries bind their handles to the thread that
made them.

Frames follow the same semantics as ``vision.capture.ScreenCapture``: BGR
``uint8``, downscaled with ``INTER_AREA`` to ``max_width``, with ``scale`` and
``origin`` set so ``Frame.to_screen`` maps back to absolute physical pixels.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol, Self

import cv2
import mss
import numpy as np
from numpy.typing import NDArray

from vision.capture import DEFAULT_MAX_WIDTH, Frame

DEFAULT_FPS_WINDOW = 2.0
_IDLE_SLEEP = 0.001
_JOIN_TIMEOUT = 5.0


class CaptureBackend(Protocol):
    """A screen grabber used from the worker thread only."""

    name: str
    origin: tuple[int, int]

    def grab(self) -> NDArray[np.uint8] | None:
        """Return the full monitor as a BGR array, or ``None`` if nothing new."""

    def close(self) -> None:
        """Release native resources."""


BackendFactory = Callable[[int | None], CaptureBackend]


@dataclass(frozen=True, eq=False)
class TimedFrame:
    """A frame plus when it was captured.

    Attributes:
        frame: the (possibly downscaled) frame.
        timestamp: ``time.monotonic()`` value when the grab finished.
        sequence: 1-based count of frames delivered by this ``LiveCapture``.
    """

    frame: Frame
    timestamp: float
    sequence: int


class _MssBackend:
    name = "mss"

    def __init__(self, monitor_index: int | None) -> None:
        self._sct = mss.MSS()
        try:
            monitors = self._sct.monitors[1:]
            if not monitors:
                raise RuntimeError("No monitors detected")
            if monitor_index is None:
                monitor = next(
                    (m for m in monitors if m.get("is_primary")), monitors[0]
                )
            elif 1 <= monitor_index <= len(monitors):
                monitor = monitors[monitor_index - 1]
            else:
                raise ValueError(
                    f"monitor_index must be between 1 and {len(monitors)}, "
                    f"got {monitor_index}"
                )
            self._box = {k: monitor[k] for k in ("left", "top", "width", "height")}
        except Exception:
            self._sct.close()
            raise
        self.origin = (self._box["left"], self._box["top"])

    def grab(self) -> NDArray[np.uint8]:
        raw = np.asarray(self._sct.grab(dict(self._box)))
        return np.ascontiguousarray(raw[:, :, :3])  # BGRA -> BGR

    def close(self) -> None:
        self._sct.close()


class _DxcamBackend:
    name = "dxcam"

    def __init__(self, monitor_index: int | None) -> None:
        import dxcam  # imported lazily: optional, Windows-only

        # Monitor geometry (for ``origin``) comes from mss; dxcam outputs are
        # 0-based while mss monitors are 1-based.
        with mss.MSS() as sct:
            monitors = sct.monitors[1:]
            if monitor_index is None:
                index = next(
                    (i for i, m in enumerate(monitors) if m.get("is_primary")), 0
                )
            elif 1 <= monitor_index <= len(monitors):
                index = monitor_index - 1
            else:
                raise ValueError(f"Invalid monitor_index {monitor_index}")
            self.origin = (monitors[index]["left"], monitors[index]["top"])

        self._camera = dxcam.create(output_idx=index, output_color="BGR")
        if self._camera is None:
            raise RuntimeError("dxcam could not create a camera")
        self._has_frame = False

    def grab(self) -> NDArray[np.uint8] | None:
        # Desktop Duplication only reports changes. Until the first frame has
        # been seen, ask for the current image even if it is not "new".
        frame = self._camera.grab(new_frame_only=self._has_frame)
        if frame is not None:
            self._has_frame = True
        return frame

    def close(self) -> None:
        self._camera.release()


def _create_mss_backend(monitor_index: int | None) -> CaptureBackend:
    return _MssBackend(monitor_index)


def _create_dxcam_backend(monitor_index: int | None) -> CaptureBackend:
    return _DxcamBackend(monitor_index)


class LiveCapture:
    """Keeps the latest screen frame fresh from a background thread.

    Use it as a context manager, or call ``start()`` and ``stop()``.
    """

    def __init__(
        self,
        monitor_index: int | None = None,
        max_width: int | None = DEFAULT_MAX_WIDTH,
        backends: Sequence[BackendFactory] | None = None,
        fps_window: float = DEFAULT_FPS_WINDOW,
    ) -> None:
        """
        Args:
            monitor_index: ``mss`` monitor index (1 = first monitor). ``None``
                selects the primary monitor.
            max_width: Frames wider than this are downscaled to it, keeping the
                aspect ratio. ``None`` disables downscaling.
            backends: Backend factories tried in order (each receives the
                monitor index). Defaults to ``dxcam`` then ``mss``.
            fps_window: Seconds of history used to measure the capture fps.

        Raises:
            ValueError: If ``max_width`` or ``fps_window`` is not positive.
        """

        if max_width is not None and max_width <= 0:
            raise ValueError(f"max_width must be positive, got {max_width}")
        if fps_window <= 0:
            raise ValueError(f"fps_window must be positive, got {fps_window}")

        self._monitor_index = monitor_index
        self._max_width = max_width
        self._factories: list[BackendFactory] = (
            list(backends)
            if backends is not None
            else [
                lambda index: _create_dxcam_backend(index),
                lambda index: _create_mss_backend(index),
            ]
        )
        self._fps_window = fps_window

        self._latest: TimedFrame | None = None
        self._sequence = 0
        self._times: deque[float] = deque()
        self._times_lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._backend_name: str | None = None
        self._error: Exception | None = None

    # ------------------------------------------------------------------ public

    def start(self) -> None:
        """Start the worker thread. Does nothing if it is already running."""

        if self.running:
            return
        self._stop_event.clear()
        self._frame_ready.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._run, name="live-capture", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the worker and wait for it. Keeps the last frame. Idempotent."""

        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        thread.join(_JOIN_TIMEOUT)
        self._thread = None

    @property
    def running(self) -> bool:
        """Whether the worker thread is alive."""

        return self._thread is not None and self._thread.is_alive()

    def latest(self) -> Frame | None:
        """Return the most recent frame without blocking, or ``None``."""

        timed = self._latest
        return None if timed is None else timed.frame

    def latest_timed(self) -> TimedFrame | None:
        """Like ``latest()`` but with the capture timestamp and sequence."""

        return self._latest

    def wait_for_frame(self, timeout: float | None = None) -> bool:
        """Block until a frame arrives after ``start()``; ``False`` on timeout."""

        return self._frame_ready.wait(timeout)

    @property
    def fps(self) -> float:
        """Capture frames per second over the last ``fps_window`` seconds."""

        return self._fps_at(time.monotonic())

    @property
    def backend_name(self) -> str | None:
        """Name of the backend currently in use (``None`` before it starts)."""

        return self._backend_name

    @property
    def error(self) -> Exception | None:
        """The last error if every backend failed, else ``None``."""

        return self._error

    # --------------------------------------------------------------- fps tracking

    def _record_frame_time(self, now: float) -> None:
        with self._times_lock:
            self._times.append(now)
            self._prune(now)

    def _fps_at(self, now: float) -> float:
        with self._times_lock:
            self._prune(now)
            if len(self._times) < 2:
                return 0.0
            span = self._times[-1] - self._times[0]
            return (len(self._times) - 1) / span if span > 0 else 0.0

    def _prune(self, now: float) -> None:
        cutoff = now - self._fps_window
        while self._times and self._times[0] < cutoff:
            self._times.popleft()

    # ------------------------------------------------------------------- worker

    def _run(self) -> None:
        last_error: Exception | None = None
        for factory in self._factories:
            if self._stop_event.is_set():
                return
            backend: CaptureBackend | None = None
            try:
                backend = factory(self._monitor_index)
                self._backend_name = backend.name
                self._loop(backend)
                return  # stopped on request
            except Exception as exc:
                last_error = exc
            finally:
                if backend is not None:
                    try:
                        backend.close()
                    except Exception:
                        pass
        self._backend_name = None
        self._error = last_error or RuntimeError("No capture backends available")

    def _loop(self, backend: CaptureBackend) -> None:
        while not self._stop_event.is_set():
            image = backend.grab()
            if image is None:
                self._stop_event.wait(_IDLE_SLEEP)
                continue
            now = time.monotonic()
            frame = self._to_frame(image, backend.origin)
            self._sequence += 1
            self._latest = TimedFrame(frame, now, self._sequence)
            self._record_frame_time(now)
            self._frame_ready.set()

    def _to_frame(self, image: NDArray[np.uint8], origin: tuple[int, int]) -> Frame:
        height, width = image.shape[:2]
        scale = 1.0
        if self._max_width is not None and width > self._max_width:
            scale = self._max_width / width
            new_height = max(1, round(height * scale))
            image = cv2.resize(
                image,
                (self._max_width, new_height),
                interpolation=cv2.INTER_AREA,
            )
        return Frame(
            image=image,
            original_size=(width, height),
            scale=scale,
            origin=origin,
        )

    # ---------------------------------------------------------------- lifecycle

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()


__all__ = [
    "BackendFactory",
    "CaptureBackend",
    "DEFAULT_FPS_WINDOW",
    "LiveCapture",
    "TimedFrame",
]
