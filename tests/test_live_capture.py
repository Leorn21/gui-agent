"""Tests for vision/live_capture.py.

Unit tests inject fake backends, so they never touch the real screen or a GPU.
The integration test grabs the real screen and only runs with
``pytest -m integration``.
"""

import threading
import time

import numpy as np
import pytest

from vision import live_capture
from vision.capture import Frame
from vision.live_capture import LiveCapture, TimedFrame

WIDTH, HEIGHT = 1920, 1080
ORIGIN = (100, 50)


class FakeBackend:
    """Backend yielding BGR frames whose first pixel encodes a counter."""

    name = "fake"

    def __init__(self, delay: float = 0.002, fail_after: int | None = None) -> None:
        self.origin = ORIGIN
        self.delay = delay
        self.fail_after = fail_after
        self.count = 0
        self.closed = False
        self.thread_id: int | None = None
        self.close_thread_id: int | None = None

    def grab(self) -> np.ndarray | None:
        self.thread_id = threading.get_ident()
        if self.fail_after is not None and self.count >= self.fail_after:
            raise RuntimeError("grab failed")
        time.sleep(self.delay)
        self.count += 1
        image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        image[0, 0] = (self.count % 256, 1, 2)
        return image

    def close(self) -> None:
        self.close_thread_id = threading.get_ident()
        self.closed = True


def _factory(backend: FakeBackend):
    def make(monitor_index: int | None) -> FakeBackend:
        backend.monitor_index = monitor_index
        return backend

    return make


def _failing_factory(error: Exception):
    def make(monitor_index: int | None):
        raise error

    return make


def _wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


# ------------------------------------------------------------------------- frames


def test_latest_is_none_before_start():
    cap = LiveCapture(backends=[_factory(FakeBackend())])
    assert cap.latest() is None
    assert cap.latest_timed() is None


def test_latest_returns_a_frame_with_capture_semantics():
    with LiveCapture(max_width=960, backends=[_factory(FakeBackend())]) as cap:
        assert cap.wait_for_frame(3.0)
        frame = cap.latest()

    assert isinstance(frame, Frame)
    assert frame.size == (960, 540)
    assert frame.original_size == (WIDTH, HEIGHT)
    assert frame.scale == pytest.approx(0.5)
    assert frame.origin == ORIGIN
    assert frame.image.dtype == np.uint8
    assert frame.to_screen(480, 270) == (ORIGIN[0] + 960, ORIGIN[1] + 540)


def test_max_width_none_keeps_full_size():
    with LiveCapture(max_width=None, backends=[_factory(FakeBackend())]) as cap:
        assert cap.wait_for_frame(3.0)
        frame = cap.latest()

    assert frame.size == (WIDTH, HEIGHT)
    assert frame.scale == 1.0


def test_downscale_uses_inter_area(monkeypatch):
    calls = []
    real_resize = live_capture.cv2.resize

    def spy(*args, **kwargs):
        calls.append(kwargs.get("interpolation"))
        return real_resize(*args, **kwargs)

    monkeypatch.setattr(live_capture.cv2, "resize", spy)
    with LiveCapture(max_width=1280, backends=[_factory(FakeBackend())]) as cap:
        assert cap.wait_for_frame(3.0)

    assert calls
    assert set(calls) == {live_capture.cv2.INTER_AREA}


@pytest.mark.parametrize("max_width", [0, -5])
def test_invalid_max_width_raises(max_width):
    with pytest.raises(ValueError):
        LiveCapture(max_width=max_width)


def test_latest_keeps_updating_and_stamps_frames():
    with LiveCapture(max_width=None, backends=[_factory(FakeBackend())]) as cap:
        assert cap.wait_for_frame(3.0)
        first = cap.latest_timed()
        assert isinstance(first, TimedFrame)
        assert _wait_for(lambda: cap.latest_timed().sequence > first.sequence + 2)
        later = cap.latest_timed()

    assert later.timestamp > first.timestamp
    assert later.frame.image[0, 0, 0] != first.frame.image[0, 0, 0]


def test_latest_never_blocks_while_the_backend_is_slow():
    slow = FakeBackend(delay=0.5)
    with LiveCapture(backends=[_factory(slow)]) as cap:
        start = time.perf_counter()
        for _ in range(1000):
            cap.latest()
        elapsed = time.perf_counter() - start

    assert elapsed < 0.2


def test_backend_is_created_and_used_inside_the_worker_thread():
    backend = FakeBackend()
    creator_threads = []

    def make(monitor_index):
        creator_threads.append(threading.get_ident())
        return backend

    with LiveCapture(backends=[make]) as cap:
        assert cap.wait_for_frame(3.0)

    assert creator_threads == [backend.thread_id]
    assert backend.thread_id != threading.get_ident()
    assert backend.close_thread_id == backend.thread_id


def test_monitor_index_is_passed_to_the_backend():
    backend = FakeBackend()
    with LiveCapture(monitor_index=2, backends=[_factory(backend)]) as cap:
        assert cap.wait_for_frame(3.0)

    assert backend.monitor_index == 2


def test_none_frames_from_the_backend_are_skipped():
    class Sparse(FakeBackend):
        def grab(self):
            self.thread_id = threading.get_ident()
            self.count += 1
            if self.count < 3:
                return None
            return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    with LiveCapture(backends=[_factory(Sparse())]) as cap:
        assert cap.wait_for_frame(3.0)
        assert cap.latest() is not None


# ---------------------------------------------------------------------------- fps


def test_fps_is_zero_before_any_frame():
    assert LiveCapture(backends=[_factory(FakeBackend())]).fps == 0.0


def test_fps_is_measured_from_recent_frames():
    with LiveCapture(backends=[_factory(FakeBackend(delay=0.02))]) as cap:
        assert _wait_for(lambda: cap.latest_timed() and cap.latest_timed().sequence > 8)
        fps = cap.fps

    assert 10 < fps < 60  # ~50 fps nominal; loose to absorb timer jitter


def test_fps_window_forgets_old_frames():
    cap = LiveCapture(backends=[_factory(FakeBackend())], fps_window=0.5)
    for t in (0.0, 0.1, 0.2):
        cap._record_frame_time(t)
    assert cap._fps_at(0.2) == pytest.approx(10.0)
    # Far later, only the single new sample remains -> not enough to measure.
    cap._record_frame_time(10.0)
    assert cap._fps_at(10.0) == 0.0


# ---------------------------------------------------------------------- fallback


def test_falls_back_to_the_next_backend_when_creation_fails():
    good = FakeBackend()
    with LiveCapture(
        backends=[_failing_factory(RuntimeError("no dxgi")), _factory(good)]
    ) as cap:
        assert cap.wait_for_frame(3.0)
        assert cap.backend_name == "fake"

    assert good.count > 0


def test_falls_back_when_the_backend_fails_while_grabbing():
    bad = FakeBackend(fail_after=2)
    good = FakeBackend()
    with LiveCapture(backends=[_factory(bad), _factory(good)]) as cap:
        assert _wait_for(lambda: good.count > 0)
        assert cap.latest() is not None

    assert bad.closed is True


def test_error_is_reported_when_every_backend_fails():
    boom = RuntimeError("boom")
    cap = LiveCapture(backends=[_failing_factory(boom), _failing_factory(boom)])
    cap.start()
    assert _wait_for(lambda: cap.error is not None)
    cap.stop()

    assert cap.latest() is None
    assert not cap.wait_for_frame(0.05)


def test_default_backends_try_dxcam_then_mss(monkeypatch):
    def broken_dxcam(monitor_index):
        raise ImportError("no dxcam")

    mss_backend = FakeBackend()
    monkeypatch.setattr(live_capture, "_create_dxcam_backend", broken_dxcam)
    monkeypatch.setattr(live_capture, "_create_mss_backend", _factory(mss_backend))

    with LiveCapture() as cap:
        assert cap.wait_for_frame(3.0)

    assert mss_backend.count > 0


# --------------------------------------------------------------------- lifecycle


def test_stop_joins_the_worker_and_closes_the_backend():
    backend = FakeBackend()
    cap = LiveCapture(backends=[_factory(backend)])
    cap.start()
    assert cap.wait_for_frame(3.0)
    worker = cap._thread
    assert worker is not None and worker.is_alive()

    cap.stop()

    assert not worker.is_alive()
    assert backend.closed is True
    assert not cap.running


def test_stop_is_idempotent_and_safe_before_start():
    cap = LiveCapture(backends=[_factory(FakeBackend())])
    cap.stop()
    cap.start()
    cap.stop()
    cap.stop()


def test_start_twice_does_not_spawn_a_second_thread():
    backend = FakeBackend()
    with LiveCapture(backends=[_factory(backend)]) as cap:
        worker = cap._thread
        cap.start()
        assert cap._thread is worker


def test_stop_keeps_the_last_frame_available():
    with LiveCapture(backends=[_factory(FakeBackend())]) as cap:
        assert cap.wait_for_frame(3.0)

    assert cap.latest() is not None


def test_stop_returns_promptly_even_with_a_slow_backend():
    cap = LiveCapture(backends=[_factory(FakeBackend(delay=0.3))])
    cap.start()
    assert cap.wait_for_frame(3.0)
    start = time.perf_counter()
    cap.stop()

    assert time.perf_counter() - start < 2.0


def test_can_restart_after_stop():
    cap = LiveCapture(backends=[_factory(FakeBackend())])
    cap.start()
    assert cap.wait_for_frame(3.0)
    cap.stop()
    cap.start()
    try:
        assert cap.running
        assert _wait_for(lambda: cap.latest_timed().sequence >= 1)
    finally:
        cap.stop()


# -------------------------------------------------------------------- integration


@pytest.mark.integration
def test_real_live_capture_reports_fps():
    with LiveCapture(max_width=1280) as cap:
        assert cap.wait_for_frame(10.0), cap.error
        time.sleep(1.0)
        frame = cap.latest()
        fps = cap.fps
        backend = cap.backend_name

    print(f"\nlive capture backend={backend} fps={fps:.1f}")
    assert frame is not None and frame.size[0] <= 1280
    assert fps > 0
