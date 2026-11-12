"""Tests for vision/diff.py.

Frames are synthetic 400x300 gray images, changed by painting rectangles or
drawing text. A region needs at least 50 changed pixels by default.
wait_for_change() gets a scripted capture function and a fake clock that
advances when it sleeps.
"""

import dataclasses

import cv2
import numpy as np
import pytest

from vision.capture import Frame
from vision.diff import (
    DEFAULT_INTERVAL,
    DEFAULT_MIN_PIXELS,
    DEFAULT_TIMEOUT,
    PIXEL_THRESHOLD,
    POLL_INTERVAL,
    DiffResult,
    FrameDiff,
    Region,
    wait_profile,
)

W, H = 400, 300
GRAY = 128


def frame(paint=(), scale=1.0, origin=(0, 0), size=(W, H)) -> Frame:
    """Gray frame with rectangles ``(x, y, w, h, value)`` painted on it."""

    width, height = size
    image = np.full((height, width, 3), GRAY, dtype=np.uint8)
    for x, y, w, h, value in paint:
        image[y : y + h, x : x + w] = value
    original = (round(width / scale), round(height / scale))
    return Frame(image=image, original_size=original, scale=scale, origin=origin)


WINDOW = (50, 40, 100, 80, 255)  # a 100x80 "window" appears: 8000 px


# ------------------------------------------------------------------- defaults


def test_defaults():
    assert PIXEL_THRESHOLD == 10
    assert DEFAULT_MIN_PIXELS == 50


def test_negative_min_pixels_raises():
    with pytest.raises(ValueError):
        FrameDiff(min_pixels=-1)


# -------------------------------------------------------------------- compare


def test_identical_frames_have_no_change():
    result = FrameDiff().compare(frame(), frame())

    assert result.changed is False
    assert result.changed_ratio == 0
    assert result.regions == ()
    assert result.bounds is None


def test_a_new_window_is_one_region_with_exact_bounds():
    result = FrameDiff().compare(frame(), frame([WINDOW]))

    assert result.changed is True
    assert result.regions == (Region(x=50, y=40, width=100, height=80),)
    assert result.bounds == Region(x=50, y=40, width=100, height=80)


def test_changed_ratio_is_the_share_of_changed_pixels():
    result = FrameDiff().compare(frame(), frame([WINDOW]))

    assert result.changed_ratio == pytest.approx(8000 / (W * H))


@pytest.mark.parametrize(
    ("value", "counts"),
    [(GRAY + PIXEL_THRESHOLD - 1, False), (GRAY + PIXEL_THRESHOLD + 1, True)],
)
def test_pixel_threshold(value, counts):
    result = FrameDiff().compare(frame(), frame([(50, 40, 100, 80, value)]))

    assert result.changed is counts


def test_blinking_text_cursor_is_ignored():
    # A 2x18 px caret (36 px) is below the minimum.
    result = FrameDiff().compare(frame(), frame([(200, 100, 2, 18, 0)]))

    assert result.changed is False


def test_typed_letters_merge_into_one_region():
    # Four 5x8 "letters" 3 px apart: each is below the minimum alone
    # (40 < 50 px), but together they are one change.
    letters = [(100 + i * 8, 150, 5, 8, 0) for i in range(4)]
    result = FrameDiff().compare(frame(), frame(letters))

    assert result.changed is True
    assert result.regions == (Region(x=100, y=150, width=29, height=8),)


def test_a_short_word_drawn_with_a_real_font_counts():
    # Regression: real glyphs are thin strokes, far fewer pixels than solid
    # blocks. A short word must still be detected on a full 1280x720 frame
    # (a 0.05% minimum, 461 px, missed it).
    full = (1280, 720)
    after = frame(size=full)
    cv2.putText(
        after.image,
        "ok",
        (60, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    result = FrameDiff().compare(frame(size=full), after)

    assert result.changed is True
    assert len(result.regions) == 1


def test_separate_changes_are_listed_largest_first_with_enclosing_bounds():
    small = (300, 220, 30, 20, 0)  # 600 px, far from the window
    result = FrameDiff().compare(frame(), frame([small, WINDOW]))

    assert result.regions == (
        Region(x=50, y=40, width=100, height=80),
        Region(x=300, y=220, width=30, height=20),
    )
    assert result.bounds == Region(x=50, y=40, width=280, height=200)


def test_regions_are_in_screen_coordinates():
    # Frame downscaled to half, captured from a region starting at (100, 50).
    before = frame(scale=0.5, origin=(100, 50))
    after = frame([WINDOW], scale=0.5, origin=(100, 50))
    result = FrameDiff().compare(before, after)

    assert result.regions == (Region(x=200, y=130, width=200, height=160),)


def test_min_pixels_is_configurable():
    result = FrameDiff(min_pixels=10_000).compare(frame(), frame([WINDOW]))  # 8000

    assert result.changed is False


@pytest.mark.parametrize(
    "other",
    [
        frame(size=(W, H + 1)),
        frame(origin=(1, 0)),
        frame(scale=0.5),
    ],
)
def test_frames_must_cover_the_same_area(other):
    with pytest.raises(ValueError):
        FrameDiff().compare(frame(), other)


def test_result_is_frozen():
    result = FrameDiff().compare(frame(), frame())

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.changed = True
    assert isinstance(result, DiffResult)


# ----------------------------------------------------------- wait_for_change


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def scripted(frames):
    """capture() returning ``frames`` in order, then repeating the last one."""

    calls = []

    def capture():
        calls.append(len(calls))
        return frames[min(len(calls) - 1, len(frames) - 1)]

    capture.calls = calls
    return capture


def wait(capture, **kwargs):
    clock = Clock()
    kwargs.setdefault("sleep", clock.sleep)
    kwargs.setdefault("clock", clock)
    result, last = FrameDiff().wait_for_change(frame(), capture, **kwargs)
    return result, last, clock


def test_waits_for_the_change_and_for_the_screen_to_settle():
    opened = frame([WINDOW])
    capture = scripted([frame(), frame(), opened, opened])
    result, last, clock = wait(capture)

    assert result.changed is True
    assert last is opened
    assert len(capture.calls) == 4  # 2 unchanged, 1 change, 1 confirming stable


def test_without_settle_returns_at_the_first_change():
    opened = frame([WINDOW])
    capture = scripted([frame(), opened, frame([WINDOW, (300, 220, 30, 20, 0)])])
    result, last, _ = wait(capture, settle=False)

    assert last is opened
    assert len(capture.calls) == 2


def test_settle_waits_out_an_animation():
    half = frame([(50, 40, 50, 80, 255)])  # window half drawn
    full = frame([WINDOW])
    capture = scripted([half, full, full])
    result, last, _ = wait(capture)

    assert last is full
    assert result.regions == (Region(x=50, y=40, width=100, height=80),)


def test_no_change_until_timeout():
    capture = scripted([frame()])
    result, _, clock = wait(capture, timeout=1.0, interval=0.1)

    assert result.changed is False
    assert clock.now == pytest.approx(1.0)
    assert len(capture.calls) == pytest.approx(11, abs=1)


def test_screen_that_never_settles_returns_at_timeout():
    flicker = [frame([(50, 40, 100, 80, v)]) for v in (200, 255) * 20]
    capture = scripted(flicker)
    result, last, clock = wait(capture, timeout=1.0, interval=0.1)

    assert result.changed is True
    assert clock.now == pytest.approx(1.0, abs=0.11)


def test_defaults_poll_every_100_ms_for_2_seconds():
    capture = scripted([frame()])
    _, _, clock = wait(capture)

    assert clock.sleeps == pytest.approx([0.1] * 20)
    assert clock.now == pytest.approx(2.0)


@pytest.mark.parametrize("kwargs", [{"timeout": -1}, {"interval": 0}])
def test_invalid_wait_settings_raise(kwargs):
    with pytest.raises(ValueError):
        wait(scripted([frame()]), **kwargs)


# ------------------------------------------------------------- wait_profile

ACTION_TYPES = (
    "CLICK",
    "DOUBLE_CLICK",
    "RIGHT_CLICK",
    "TYPE",
    "PRESS_KEY",
    "DRAG",
    "SCROLL",
    "WAIT",
)


def test_poll_interval_is_30_ms():
    assert POLL_INTERVAL == pytest.approx(0.03)


@pytest.mark.parametrize("action_type", ACTION_TYPES)
def test_every_action_type_has_a_profile_polling_every_30_ms(action_type):
    profile = wait_profile(action_type)

    assert profile["interval"] == pytest.approx(POLL_INTERVAL)
    assert 0 < profile["timeout"] <= 3.0


@pytest.mark.parametrize("action_type", ["PRESS_KEY", "TYPE", "SCROLL"])
def test_keys_typing_and_scroll_have_short_caps(action_type):
    assert wait_profile(action_type)["timeout"] <= 1.0


@pytest.mark.parametrize("action_type", ["CLICK", "DOUBLE_CLICK"])
def test_clicks_allow_longer_waits_for_programs_to_open(action_type):
    assert wait_profile(action_type)["timeout"] > wait_profile("PRESS_KEY")["timeout"]


def test_unknown_action_type_gets_the_default_settings():
    assert wait_profile("NOT_AN_ACTION") == {
        "timeout": DEFAULT_TIMEOUT,
        "interval": DEFAULT_INTERVAL,
    }


def test_profile_is_a_fresh_dict_each_call():
    profile = wait_profile("CLICK")
    profile["timeout"] = 99.0

    assert wait_profile("CLICK")["timeout"] != 99.0


def test_profile_keys_are_wait_for_change_arguments():
    capture = scripted([frame()])
    result, _, _ = wait(capture, **wait_profile("PRESS_KEY"))

    assert result.changed is False


def test_key_press_without_change_waits_well_under_2_seconds():
    capture = scripted([frame()])
    _, _, clock = wait(capture, **wait_profile("PRESS_KEY"))

    assert clock.now < 1.0
    assert clock.now < DEFAULT_TIMEOUT


def test_key_press_polls_every_30_ms():
    capture = scripted([frame()])
    _, _, clock = wait(capture, **wait_profile("PRESS_KEY"))

    assert clock.sleeps
    assert clock.sleeps == pytest.approx([POLL_INTERVAL] * len(clock.sleeps))
