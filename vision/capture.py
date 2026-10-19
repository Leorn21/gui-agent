"""Screen capture for the GUI agent.

Grabs the screen (or a region of it) with ``mss`` and returns a ``Frame``: a
BGR image downscaled to save VLM tokens, plus the metadata needed to map
coordinates on that image back to absolute screen pixels.

All coordinates are physical pixels as reported by ``mss``. Converting them to
the logical pixels some input libraries expect on High-DPI displays is the
controller's job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import TracebackType
from typing import Self

import cv2
import mss
import numpy as np
from numpy.typing import NDArray
from PIL import Image

DEFAULT_MAX_WIDTH = 1280


@dataclass(frozen=True, eq=False)
class Frame:
    """A captured image and how it relates to the real screen.

    Attributes:
        image: BGR ``uint8`` array of shape (height, width, 3), possibly downscaled.
        original_size: (width, height) of the captured area before downscaling.
        scale: ``image width / original width`` (1.0 when not downscaled).
        origin: absolute screen position (x, y) of the captured area's top-left corner.
    """

    image: NDArray[np.uint8]
    original_size: tuple[int, int]
    scale: float
    origin: tuple[int, int]

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of ``image``."""

        height, width = self.image.shape[:2]
        return width, height

    def to_pil(self) -> Image.Image:
        """Return the image as an RGB PIL image."""

        return Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))

    def to_screen(self, x: int, y: int) -> tuple[int, int]:
        """Map a point on ``image`` to absolute screen pixels.

        Undoes the downscale and adds ``origin``, rounding half up to the
        nearest pixel.

        Raises:
            ValueError: If the point lies outside ``image``.
        """

        width, height = self.size
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Point ({x}, {y}) is outside the {width}x{height} frame")
        return (
            self.origin[0] + math.floor(x / self.scale + 0.5),
            self.origin[1] + math.floor(y / self.scale + 0.5),
        )


class ScreenCapture:
    """Captures one monitor, or a region of it, as ``Frame`` objects.

    Holds a single ``mss`` instance for its whole lifetime; use it as a context
    manager or call ``close()`` when done.
    """

    def __init__(
        self,
        monitor_index: int | None = None,
        max_width: int | None = DEFAULT_MAX_WIDTH,
    ) -> None:
        """
        Args:
            monitor_index: ``mss`` monitor index (1 = first monitor). ``None``
                selects the primary monitor.
            max_width: Frames wider than this are downscaled to it, keeping the
                aspect ratio. ``None`` disables downscaling.

        Raises:
            ValueError: If ``max_width`` is not positive or ``monitor_index``
                does not name a monitor.
        """

        if max_width is not None and max_width <= 0:
            raise ValueError(f"max_width must be positive, got {max_width}")

        self._max_width = max_width
        self._sct = mss.MSS()
        self._closed = False
        try:
            self._monitor = self._resolve_monitor(monitor_index)
        except Exception:
            self.close()
            raise

    def _resolve_monitor(self, monitor_index: int | None) -> dict[str, int]:
        # monitors[0] is the union of all monitors; real ones start at 1.
        monitors = self._sct.monitors[1:]
        if not monitors:
            raise RuntimeError("No monitors detected")

        if monitor_index is None:
            monitor = next((m for m in monitors if m.get("is_primary")), monitors[0])
        elif 1 <= monitor_index <= len(monitors):
            monitor = monitors[monitor_index - 1]
        else:
            raise ValueError(
                f"monitor_index must be between 1 and {len(monitors)}, "
                f"got {monitor_index}"
            )

        return {key: monitor[key] for key in ("left", "top", "width", "height")}

    def capture(self) -> Frame:
        """Capture the whole monitor."""

        return self._grab(dict(self._monitor))

    def capture_region(self, x: int, y: int, width: int, height: int) -> Frame:
        """Capture a rectangle given relative to the monitor's top-left corner.

        Raises:
            ValueError: If the rectangle is empty or not fully inside the monitor.
        """

        if width <= 0 or height <= 0:
            raise ValueError(f"Region size must be positive, got {width}x{height}")
        if (
            x < 0
            or y < 0
            or x + width > self._monitor["width"]
            or y + height > self._monitor["height"]
        ):
            raise ValueError(
                f"Region ({x}, {y}, {width}x{height}) is outside the "
                f"{self._monitor['width']}x{self._monitor['height']} monitor"
            )

        return self._grab(
            {
                "left": self._monitor["left"] + x,
                "top": self._monitor["top"] + y,
                "width": width,
                "height": height,
            }
        )

    def _grab(self, box: dict[str, int]) -> Frame:
        if self._closed:
            raise RuntimeError("ScreenCapture is closed")

        raw = np.asarray(self._sct.grab(box))
        image = np.ascontiguousarray(raw[:, :, :3])  # BGRA -> BGR
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
            origin=(box["left"], box["top"]),
        )

    def close(self) -> None:
        """Release the ``mss`` instance. Safe to call more than once."""

        if not self._closed:
            self._sct.close()
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


__all__ = [
    "DEFAULT_MAX_WIDTH",
    "Frame",
    "ScreenCapture",
]
