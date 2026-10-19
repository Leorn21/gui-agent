"""Tests for vision/capture.py.

Unit tests replace ``mss.MSS`` with a fake backed by a synthetic virtual
desktop, so they never touch the real screen. Tests marked ``integration``
grab the real primary monitor and only run with ``pytest -m integration``.
"""

import dataclasses

import cv2
import numpy as np
import pytest
from PIL import Image

from vision import capture
from vision.capture import Frame, ScreenCapture

# Virtual desktop: a secondary monitor to the LEFT of the primary one, listed
# first so the primary is not at index 1.
SECONDARY = {"left": -1280, "top": 0, "width": 1280, "height": 1024}
PRIMARY = {"left": 0, "top": 0, "width": 1920, "height": 1080, "is_primary": True}
ALL_MONITORS = {"left": -1280, "top": 0, "width": 3200, "height": 1080}


def _virtual_desktop() -> np.ndarray:
    """BGRA desktop where B = x % 256, G = y % 256, R = 7, A = 255 (absolute coords)."""

    height, width = ALL_MONITORS["height"], ALL_MONITORS["width"]
    xs = (np.arange(width) + ALL_MONITORS["left"]) % 256
    ys = (np.arange(height) + ALL_MONITORS["top"]) % 256
    desktop = np.empty((height, width, 4), dtype=np.uint8)
    desktop[..., 0] = xs[np.newaxis, :]
    desktop[..., 1] = ys[:, np.newaxis]
    desktop[..., 2] = 7
    desktop[..., 3] = 255
    return desktop


class FakeMSS:
    """Minimal stand-in for ``mss.MSS``."""

    instances: list["FakeMSS"] = []

    def __init__(self) -> None:
        self.monitors = [ALL_MONITORS, SECONDARY, PRIMARY]
        self.grabs: list[dict] = []
        self.closed = False
        self._desktop = _virtual_desktop()
        FakeMSS.instances.append(self)

    def grab(self, box: dict) -> np.ndarray:
        self.grabs.append(dict(box))
        x0 = box["left"] - ALL_MONITORS["left"]
        y0 = box["top"] - ALL_MONITORS["top"]
        return self._desktop[y0 : y0 + box["height"], x0 : x0 + box["width"]].copy()

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def fake_mss(monkeypatch):
    FakeMSS.instances = []
    monkeypatch.setattr(capture.mss, "MSS", FakeMSS)
    return FakeMSS


def _bgr_at(frame: Frame, x: int, y: int) -> tuple[int, int, int]:
    return tuple(int(v) for v in frame.image[y, x])


# --------------------------------------------------------------------------- Frame


def _make_frame(**overrides) -> Frame:
    params = dict(
        image=np.zeros((50, 100, 3), dtype=np.uint8),
        original_size=(200, 100),
        scale=0.5,
        origin=(10, 20),
    )
    params.update(overrides)
    return Frame(**params)


def test_frame_size_is_width_height_of_image():
    assert _make_frame().size == (100, 50)


def test_frame_is_frozen():
    frame = _make_frame()
    with pytest.raises(dataclasses.FrozenInstanceError):
        frame.scale = 1.0


def test_frame_to_pil_converts_bgr_to_rgb():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0] = (255, 0, 0)  # pure blue in BGR
    pil = _make_frame(image=image, original_size=(2, 2), scale=1.0).to_pil()

    assert isinstance(pil, Image.Image)
    assert pil.mode == "RGB"
    assert pil.size == (2, 2)
    assert pil.getpixel((0, 0)) == (0, 0, 255)


def test_frame_to_screen_undoes_scale_and_adds_origin():
    frame = _make_frame()  # scale 0.5, origin (10, 20)
    assert frame.to_screen(0, 0) == (10, 20)
    assert frame.to_screen(30, 40) == (10 + 60, 20 + 80)


def test_frame_to_screen_rounds_to_nearest_pixel():
    frame = _make_frame(scale=2 / 3, origin=(0, 0))
    assert frame.to_screen(10, 10) == (15, 15)
    assert frame.to_screen(1, 1) == (2, 2)  # 1.5 rounds up


@pytest.mark.parametrize("point", [(-1, 0), (0, -1), (100, 0), (0, 50)])
def test_frame_to_screen_rejects_points_outside_image(point):
    with pytest.raises(ValueError):
        _make_frame().to_screen(*point)


# ------------------------------------------------------------------ ScreenCapture


def test_default_captures_the_primary_monitor_by_flag():
    with ScreenCapture(max_width=None) as cap:
        frame = cap.capture()

    assert FakeMSS.instances[0].grabs == [
        {k: PRIMARY[k] for k in ("left", "top", "width", "height")}
    ]
    assert frame.original_size == (1920, 1080)
    assert frame.origin == (0, 0)


def test_monitor_index_selects_another_monitor():
    with ScreenCapture(monitor_index=1, max_width=None) as cap:
        frame = cap.capture()

    assert frame.original_size == (1280, 1024)
    assert frame.origin == (-1280, 0)
    # Top-left pixel comes from absolute x=-1280 -> B = -1280 % 256 = 0
    assert _bgr_at(frame, 0, 0) == (0, 0, 7)


@pytest.mark.parametrize("index", [0, 3, -1])
def test_invalid_monitor_index_raises(index):
    with pytest.raises(ValueError):
        ScreenCapture(monitor_index=index)


def test_capture_returns_bgr_uint8_without_alpha():
    with ScreenCapture(max_width=None) as cap:
        frame = cap.capture()

    assert frame.image.shape == (1080, 1920, 3)
    assert frame.image.dtype == np.uint8
    assert _bgr_at(frame, 5, 3) == (5, 3, 7)


def test_capture_downscales_to_max_width_keeping_aspect_ratio():
    with ScreenCapture(max_width=1280) as cap:
        frame = cap.capture()

    assert frame.size == (1280, 720)
    assert frame.original_size == (1920, 1080)
    assert frame.scale == pytest.approx(1280 / 1920)


def test_capture_default_max_width_is_1280():
    with ScreenCapture() as cap:
        frame = cap.capture()

    assert frame.size[0] == 1280


def test_capture_never_upscales_small_screens():
    with ScreenCapture(monitor_index=1, max_width=4000) as cap:
        frame = cap.capture()

    assert frame.size == (1280, 1024)
    assert frame.scale == 1.0


def test_downscaled_coordinates_map_back_to_screen():
    with ScreenCapture(max_width=960) as cap:  # exactly half of 1920
        frame = cap.capture()

    assert frame.to_screen(480, 270) == (960, 540)


def test_downscale_uses_inter_area(monkeypatch):
    calls = []
    real_resize = cv2.resize

    def spy(*args, **kwargs):
        calls.append(kwargs.get("interpolation"))
        return real_resize(*args, **kwargs)

    monkeypatch.setattr(capture.cv2, "resize", spy)
    with ScreenCapture(max_width=1280) as cap:
        cap.capture()

    assert calls == [cv2.INTER_AREA]


@pytest.mark.parametrize("max_width", [0, -10])
def test_invalid_max_width_raises(max_width):
    with pytest.raises(ValueError):
        ScreenCapture(max_width=max_width)


# ------------------------------------------------------------------ capture_region


def test_capture_region_is_relative_to_the_monitor():
    with ScreenCapture(monitor_index=1, max_width=None) as cap:
        frame = cap.capture_region(x=100, y=50, width=200, height=80)

    assert FakeMSS.instances[0].grabs == [
        {"left": -1280 + 100, "top": 50, "width": 200, "height": 80}
    ]
    assert frame.origin == (-1180, 50)
    assert frame.original_size == (200, 80)
    assert _bgr_at(frame, 0, 0) == ((-1180) % 256, 50, 7)


def test_capture_region_also_respects_max_width():
    with ScreenCapture(max_width=100) as cap:
        frame = cap.capture_region(x=0, y=0, width=400, height=200)

    assert frame.size == (100, 50)
    assert frame.to_screen(50, 25) == (200, 100)


def test_capture_region_with_offset_and_downscale_maps_back_to_screen():
    # e.g. a browser window at (600, 200) that does not fill the screen
    with ScreenCapture(max_width=400) as cap:
        frame = cap.capture_region(x=600, y=200, width=800, height=600)

    assert frame.size == (400, 300)
    assert frame.origin == (600, 200)
    assert frame.scale == pytest.approx(0.5)
    assert frame.to_screen(100, 50) == (800, 300)


@pytest.mark.parametrize(
    "region",
    [
        dict(x=-1, y=0, width=10, height=10),
        dict(x=0, y=-1, width=10, height=10),
        dict(x=0, y=0, width=0, height=10),
        dict(x=0, y=0, width=10, height=0),
        dict(x=1900, y=0, width=50, height=10),  # spills past right edge
        dict(x=0, y=1070, width=10, height=20),  # spills past bottom edge
    ],
)
def test_capture_region_outside_monitor_raises(region):
    with ScreenCapture() as cap:
        with pytest.raises(ValueError):
            cap.capture_region(**region)


# ---------------------------------------------------------------------- lifecycle


def test_reuses_a_single_mss_instance_across_captures():
    with ScreenCapture() as cap:
        cap.capture()
        cap.capture()

    assert len(FakeMSS.instances) == 1


def test_context_manager_closes_mss():
    with ScreenCapture() as cap:
        cap.capture()

    assert FakeMSS.instances[0].closed is True


def test_close_is_idempotent():
    cap = ScreenCapture()
    cap.capture()
    cap.close()
    cap.close()

    assert FakeMSS.instances[0].closed is True


# -------------------------------------------------------------------- integration


@pytest.mark.integration
def test_real_capture_of_primary_monitor(monkeypatch):
    monkeypatch.undo()  # use the real mss backend

    with ScreenCapture(max_width=1280) as cap:
        frame = cap.capture()

    assert frame.image.ndim == 3 and frame.image.shape[2] == 3
    assert frame.size[0] <= 1280
    assert frame.original_size[0] >= frame.size[0]
