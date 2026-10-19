"""Tests for controller/base.py and controller/os_controller.py.

Unit tests inject fake pynput mouse/keyboard controllers, a fake safety switch
and a recording sleep, so nothing moves the real mouse and nothing waits. The
integration tests move the real cursor and are opt-in (``-m integration``).
"""

import pytest
from pynput import keyboard, mouse

from controller.base import BaseController
from controller.os_controller import OSController
from controller.safety import PausedError

BOUNDS = (0, 0, 1920, 1080)  # left, top, width, height of the virtual screen


class FakeMouse:
    def __init__(self, position=(100, 100)) -> None:
        self._position = position
        self.positions: list[tuple[int, int]] = []
        self.events: list[tuple] = []

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = tuple(value)
        self.positions.append(self._position)

    def press(self, button):
        self.events.append(("press", button, self._position))

    def release(self, button):
        self.events.append(("release", button, self._position))

    def click(self, button, count=1):
        self.events.append(("click", button, count, self._position))

    def scroll(self, dx, dy):
        self.events.append(("scroll", dx, dy, self._position))


class FakeKeyboard:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def press(self, key):
        self.events.append(("press", key))

    def release(self, key):
        self.events.append(("release", key))

    def type(self, text):
        self.events.append(("type", text))


class FakeSafety:
    """check() raises ``error`` once it has been called ``fail_after`` times."""

    def __init__(self, fail_after=None, error=PausedError) -> None:
        self.calls = 0
        self.fail_after = fail_after
        self.error = error

    def check(self):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise self.error("stop")


@pytest.fixture
def parts():
    return {
        "mouse": FakeMouse(),
        "keyboard": FakeKeyboard(),
        "sleeps": [],
        "dpi_calls": [],
    }


@pytest.fixture
def make(parts):
    def build(**kwargs):
        kwargs.setdefault("mouse_controller", parts["mouse"])
        kwargs.setdefault("keyboard_controller", parts["keyboard"])
        kwargs.setdefault("sleep", parts["sleeps"].append)
        kwargs.setdefault("screen_bounds", BOUNDS)
        kwargs.setdefault("make_dpi_aware", lambda: parts["dpi_calls"].append(True))
        return OSController(**kwargs)

    return build


def typed(kb: FakeKeyboard) -> str:
    return "".join(e[1] for e in kb.events if e[0] == "type")


# -------------------------------------------------------------- BaseController


def test_base_controller_is_abstract():
    with pytest.raises(TypeError):
        BaseController()


def test_subclass_must_implement_every_action():
    class Partial(BaseController):
        def click(self, x, y, button="left"):
            pass

    with pytest.raises(TypeError):
        Partial()


def test_os_controller_is_a_base_controller(make):
    assert isinstance(make(), BaseController)


# ---------------------------------------------------------------- construction


def test_makes_the_process_dpi_aware_once(make, parts):
    make()
    assert parts["dpi_calls"] == [True]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"move_duration": -0.1},
        {"move_steps": 0},
        {"type_interval": -0.01},
    ],
)
def test_invalid_settings_raise(make, kwargs):
    with pytest.raises(ValueError):
        make(**kwargs)


# --------------------------------------------------------------------- movement


def test_click_moves_along_a_trajectory_then_clicks(make, parts):
    make(move_steps=10).click(500, 300)
    m = parts["mouse"]

    assert len(m.positions) == 10
    assert m.positions[-1] == (500, 300)
    # every intermediate point lies between start and target
    assert all(100 <= x <= 500 and 100 <= y <= 300 for x, y in m.positions)
    assert m.events == [("click", mouse.Button.left, 1, (500, 300))]


def test_move_duration_is_spread_over_the_steps(make, parts):
    make(move_duration=0.2, move_steps=10).click(500, 300)

    assert sum(parts["sleeps"]) == pytest.approx(0.2)


def test_zero_duration_jumps_directly(make, parts):
    make(move_duration=0).click(500, 300)

    assert parts["mouse"].positions == [(500, 300)]
    assert sum(parts["sleeps"]) == 0


def test_default_move_duration_is_two_tenths(make, parts):
    make().click(500, 300)

    assert sum(parts["sleeps"]) == pytest.approx(0.2)


def test_right_and_middle_buttons(make, parts):
    ctl = make()
    ctl.click(10, 10, button="right")
    ctl.click(10, 10, button="middle")

    buttons = [e[1] for e in parts["mouse"].events]
    assert buttons == [mouse.Button.right, mouse.Button.middle]


def test_invalid_button_raises(make):
    with pytest.raises(ValueError):
        make().click(10, 10, button="side")


def test_double_click(make, parts):
    make().double_click(200, 200)

    assert parts["mouse"].events == [("click", mouse.Button.left, 2, (200, 200))]


@pytest.mark.parametrize("point", [(-1, 0), (0, -1), (1920, 0), (0, 1080)])
def test_click_outside_the_screen_raises_without_moving(make, parts, point):
    with pytest.raises(ValueError):
        make().click(*point)

    assert parts["mouse"].positions == []
    assert parts["mouse"].events == []


def test_negative_coordinates_allowed_on_a_left_monitor(make, parts):
    # Virtual screen with a secondary monitor to the left of the primary.
    make(screen_bounds=(-1280, 0, 3200, 1080)).click(-640, 500)

    assert parts["mouse"].positions[-1] == (-640, 500)


# ----------------------------------------------------------------------- typing


def test_type_text_types_each_character_including_unicode(make, parts):
    make().type_text("Hola ñandú, ¿qué tal?")

    assert typed(parts["keyboard"]) == "Hola ñandú, ¿qué tal?"
    assert all(len(e[1]) == 1 for e in parts["keyboard"].events)


def test_type_interval_between_characters(make, parts):
    make(type_interval=0.02).type_text("abcd")

    assert parts["sleeps"] == pytest.approx([0.02, 0.02, 0.02, 0.02])


def test_default_type_interval_is_20_ms(make, parts):
    make().type_text("ab")

    assert parts["sleeps"] == pytest.approx([0.02, 0.02])


def test_type_empty_text_raises(make):
    with pytest.raises(ValueError):
        make().type_text("")


# ---------------------------------------------------------------------- keys


def test_press_single_key(make, parts):
    make().press_key("enter")
    (enter,) = keyboard.HotKey.parse("<enter>")

    assert parts["keyboard"].events == [("press", enter), ("release", enter)]


def test_press_combo_presses_in_order_and_releases_in_reverse(make, parts):
    make().press_key("ctrl+s")
    ctrl, s = keyboard.HotKey.parse("<ctrl>+s")

    assert parts["keyboard"].events == [
        ("press", ctrl),
        ("press", s),
        ("release", s),
        ("release", ctrl),
    ]


def test_press_unknown_key_raises(make, parts):
    with pytest.raises(ValueError):
        make().press_key("ctrl+notakey")

    assert parts["keyboard"].events == []


@pytest.mark.parametrize("combo", ["ctrl+alt+q", "Ctrl + Alt + R"])
def test_safety_hotkeys_are_refused(make, parts, combo):
    with pytest.raises(ValueError):
        make().press_key(combo)

    assert parts["keyboard"].events == []


def test_custom_reserved_hotkeys(make, parts):
    ctl = make(reserved_hotkeys=("ctrl+shift+p",))

    with pytest.raises(ValueError):
        ctl.press_key("ctrl+shift+p")
    ctl.press_key("ctrl+alt+q")  # no longer reserved

    assert parts["keyboard"].events  # the second combo was pressed


# ------------------------------------------------------------------------- drag


def test_drag_presses_moves_and_releases(make, parts):
    make().drag(100, 100, 400, 250, duration=0.5)
    m = parts["mouse"]

    assert m.events[0] == ("press", mouse.Button.left, (100, 100))
    assert m.events[-1] == ("release", mouse.Button.left, (400, 250))
    assert m.positions[-1] == (400, 250)


def test_drag_uses_its_own_duration(make, parts):
    make(move_duration=0).drag(100, 100, 400, 250, duration=0.5)

    assert sum(parts["sleeps"]) == pytest.approx(0.5)


def test_drag_outside_the_screen_raises(make, parts):
    with pytest.raises(ValueError):
        make().drag(100, 100, 5000, 250, duration=0.5)

    assert parts["mouse"].events == []


# ------------------------------------------------------------------------ scroll


def test_scroll_positive_amount_scrolls_down_at_the_point(make, parts):
    make().scroll(800, 600, 3)

    assert parts["mouse"].events == [("scroll", 0, -3, (800, 600))]


def test_scroll_negative_amount_scrolls_up(make, parts):
    make().scroll(800, 600, -2)

    assert parts["mouse"].events == [("scroll", 0, 2, (800, 600))]


def test_scroll_zero_raises(make):
    with pytest.raises(ValueError):
        make().scroll(800, 600, 0)


# ------------------------------------------------------------------------ safety


def test_actions_work_without_a_safety_switch(make, parts):
    make(safety=None).type_text("ok")

    assert typed(parts["keyboard"]) == "ok"


def test_paused_before_click_does_nothing(make, parts):
    with pytest.raises(PausedError):
        make(safety=FakeSafety(fail_after=0)).click(500, 300)

    assert parts["mouse"].events == []
    assert parts["mouse"].positions == []


def test_pause_cuts_typing_between_characters(make, parts):
    with pytest.raises(PausedError):
        make(safety=FakeSafety(fail_after=3)).type_text("abcdef")

    assert typed(parts["keyboard"]) == "abc"


def test_movement_checks_safety_between_steps(make, parts):
    safety = FakeSafety(fail_after=4)
    with pytest.raises(PausedError):
        make(safety=safety, move_steps=10).click(500, 300)

    assert len(parts["mouse"].positions) == 4
    assert parts["mouse"].events == []  # never clicked


def test_abort_during_drag_releases_the_button(make, parts):
    with pytest.raises(InterruptedError):
        # move_duration=0: reaching the start is one step, so the abort lands
        # while the button is held down.
        make(
            move_duration=0, safety=FakeSafety(fail_after=5, error=InterruptedError)
        ).drag(100, 100, 400, 250, duration=0.5)

    events = parts["mouse"].events
    assert events[0][:2] == ("press", mouse.Button.left)
    assert events[-1][:2] == ("release", mouse.Button.left)


def test_interrupted_combo_releases_held_keys(make, parts):
    class FailingKeyboard(FakeKeyboard):
        def press(self, key):
            super().press(key)
            if len(self.events) == 2:  # fails while pressing the second key
                raise RuntimeError("driver error")

    kb = FailingKeyboard()
    with pytest.raises(RuntimeError):
        make(keyboard_controller=kb).press_key("ctrl+shift+s")

    pressed = [e[1] for e in kb.events if e[0] == "press"]
    released = [e[1] for e in kb.events if e[0] == "release"]
    assert set(released) == set(pressed)


# ------------------------------------------------------------------- integration


@pytest.mark.integration
def test_real_cursor_reaches_physical_screen_corner():
    # Moves the real mouse. After DPI awareness, pynput and mss must agree on
    # the bottom-right pixel of the primary monitor.
    import mss

    with mss.MSS() as sct:
        primary = next(m for m in sct.monitors[1:] if m.get("is_primary"))
    corner = (
        primary["left"] + primary["width"] - 2,
        primary["top"] + primary["height"] - 2,
    )

    ctl = OSController(move_duration=0)
    start = ctl.position
    try:
        ctl.move_to(*corner)
        assert ctl.position == corner
    finally:
        ctl.move_to(*start)
