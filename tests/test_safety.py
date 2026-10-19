"""Tests for controller/safety.py.

Unit tests inject a fake hotkey listener and call its callbacks directly to
simulate key presses, plus a fake clock for the debounce. The integration test
starts the real pynput listener without sending any keystrokes.
"""

import threading

import pytest

from controller.safety import (
    DEFAULT_PAUSE_HOTKEY,
    DEFAULT_RESUME_HOTKEY,
    EmergencyKillSwitch,
    PausedError,
    SafetyState,
    to_pynput_hotkey,
)


class FakeListener:
    """Stand-in for pynput's GlobalHotKeys: keeps the hotkey -> callback map."""

    def __init__(self, hotkeys: dict) -> None:
        self.hotkeys = hotkeys
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def press(self, combo: str) -> None:
        self.hotkeys[combo]()


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def make_switch(clock):
    listeners: list[FakeListener] = []

    def factory(hotkeys):
        listener = FakeListener(hotkeys)
        listeners.append(listener)
        return listener

    def make(**kwargs):
        kwargs.setdefault("listener_factory", factory)
        kwargs.setdefault("clock", clock)
        switch = EmergencyKillSwitch(**kwargs)
        switch.start()
        return switch, listeners[-1]

    return make


PAUSE = "<ctrl>+<alt>+q"
RESUME = "<ctrl>+<alt>+r"


# ------------------------------------------------------------------ hotkey parsing


def test_defaults_are_ctrl_alt_q_and_r():
    assert DEFAULT_PAUSE_HOTKEY == "ctrl+alt+q"
    assert DEFAULT_RESUME_HOTKEY == "ctrl+alt+r"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ctrl+alt+q", "<ctrl>+<alt>+q"),
        ("Ctrl + Alt + Q", "<ctrl>+<alt>+q"),
        ("ctrl+shift+f12", "<ctrl>+<shift>+<f12>"),
    ],
)
def test_to_pynput_hotkey(text, expected):
    assert to_pynput_hotkey(text) == expected


@pytest.mark.parametrize("text", ["", "ctrl+", "ctrl+notakey", "+q"])
def test_invalid_hotkey_raises(text):
    with pytest.raises(ValueError):
        to_pynput_hotkey(text)


def test_pause_and_resume_hotkeys_must_differ(make_switch):
    with pytest.raises(ValueError):
        make_switch(pause_hotkey="ctrl+alt+q", resume_hotkey="Ctrl+Alt+Q")


def test_listener_registers_both_hotkeys(make_switch):
    _, listener = make_switch()

    assert set(listener.hotkeys) == {PAUSE, RESUME}
    assert listener.started is True


def test_custom_hotkeys_are_registered(make_switch):
    _, listener = make_switch(pause_hotkey="ctrl+shift+p", resume_hotkey="ctrl+shift+o")

    assert set(listener.hotkeys) == {"<ctrl>+<shift>+p", "<ctrl>+<shift>+o"}


# ---------------------------------------------------------------- state machine


def test_starts_running(make_switch):
    switch, _ = make_switch()

    assert switch.state is SafetyState.RUNNING
    assert not switch.is_paused()
    assert not switch.is_aborted()
    switch.check()  # does not raise


def test_pause_hotkey_pauses(make_switch):
    switch, listener = make_switch()
    listener.press(PAUSE)

    assert switch.state is SafetyState.PAUSED
    assert switch.is_paused()


def test_resume_hotkey_resumes(make_switch, clock):
    switch, listener = make_switch()
    listener.press(PAUSE)
    clock.advance(1)
    listener.press(RESUME)

    assert switch.state is SafetyState.RUNNING


def test_pause_hotkey_twice_aborts(make_switch, clock):
    switch, listener = make_switch()
    listener.press(PAUSE)
    clock.advance(1)
    listener.press(PAUSE)

    assert switch.state is SafetyState.ABORTED
    assert switch.is_aborted()


def test_abort_is_latched(make_switch, clock):
    switch, listener = make_switch()
    listener.press(PAUSE)
    clock.advance(1)
    listener.press(PAUSE)
    clock.advance(1)
    listener.press(RESUME)
    clock.advance(1)
    listener.press(PAUSE)

    assert switch.state is SafetyState.ABORTED


def test_resume_while_running_does_nothing(make_switch):
    switch, listener = make_switch()
    listener.press(RESUME)

    assert switch.state is SafetyState.RUNNING


def test_held_pause_hotkey_does_not_abort(make_switch, clock):
    # Key auto-repeat fires the hotkey again a few ms later: must stay paused.
    switch, listener = make_switch()
    listener.press(PAUSE)
    clock.advance(0.05)
    listener.press(PAUSE)
    clock.advance(0.3)
    listener.press(PAUSE)

    assert switch.state is SafetyState.PAUSED


def test_debounce_window_is_configurable(make_switch, clock):
    switch, listener = make_switch(debounce=2.0)
    listener.press(PAUSE)
    clock.advance(1.5)
    listener.press(PAUSE)
    assert switch.state is SafetyState.PAUSED

    clock.advance(1.0)
    listener.press(PAUSE)
    assert switch.state is SafetyState.ABORTED


def test_programmatic_controls(make_switch):
    switch, _ = make_switch()

    switch.pause()
    assert switch.state is SafetyState.PAUSED
    switch.resume()
    assert switch.state is SafetyState.RUNNING
    switch.abort()
    assert switch.state is SafetyState.ABORTED
    switch.resume()
    assert switch.state is SafetyState.ABORTED


def test_abort_from_running(make_switch):
    switch, _ = make_switch()
    switch.abort()

    assert switch.is_aborted()


# ------------------------------------------------------------------------ check


def test_check_raises_paused_error_while_paused(make_switch):
    switch, _ = make_switch()
    switch.pause()

    with pytest.raises(PausedError):
        switch.check()


def test_check_raises_interrupted_error_when_aborted(make_switch):
    switch, _ = make_switch()
    switch.abort()

    with pytest.raises(InterruptedError):
        switch.check()


def test_paused_error_is_not_an_interrupted_error():
    # The FSM must tell "re-perceive after resume" apart from "stop for good".
    assert not issubclass(PausedError, InterruptedError)


# ------------------------------------------------------------ wait_while_paused


def test_wait_returns_immediately_when_running(make_switch):
    switch, _ = make_switch()
    switch.wait_while_paused(timeout=0.01)


def test_wait_blocks_until_resumed(make_switch):
    switch, _ = make_switch()
    switch.pause()
    done = threading.Event()

    def waiter():
        switch.wait_while_paused()
        done.set()

    thread = threading.Thread(target=waiter, daemon=True)
    thread.start()
    assert not done.wait(0.1)  # still blocked

    switch.resume()
    assert done.wait(1.0)


def test_wait_raises_when_aborted_during_pause(make_switch):
    switch, _ = make_switch()
    switch.pause()
    errors = []

    def waiter():
        try:
            switch.wait_while_paused()
        except InterruptedError as exc:
            errors.append(exc)

    thread = threading.Thread(target=waiter, daemon=True)
    thread.start()
    switch.abort()
    thread.join(1.0)

    assert len(errors) == 1


def test_wait_raises_when_already_aborted(make_switch):
    switch, _ = make_switch()
    switch.abort()

    with pytest.raises(InterruptedError):
        switch.wait_while_paused(timeout=0.01)


def test_wait_timeout_returns_false_if_still_paused(make_switch):
    switch, _ = make_switch()
    switch.pause()

    assert switch.wait_while_paused(timeout=0.05) is False


# -------------------------------------------------------------------- callbacks


def test_on_state_change_reports_each_transition(make_switch, clock):
    seen = []
    _, listener = make_switch(on_state_change=seen.append)

    listener.press(PAUSE)
    clock.advance(1)
    listener.press(RESUME)
    clock.advance(1)
    listener.press(PAUSE)
    clock.advance(1)
    listener.press(PAUSE)

    assert seen == [
        SafetyState.PAUSED,
        SafetyState.RUNNING,
        SafetyState.PAUSED,
        SafetyState.ABORTED,
    ]


def test_on_state_change_not_called_for_no_ops(make_switch):
    seen = []
    switch, listener = make_switch(on_state_change=seen.append)

    listener.press(RESUME)  # already running
    switch.abort()
    switch.abort()  # already aborted

    assert seen == [SafetyState.ABORTED]


def test_failing_callback_does_not_break_the_switch(make_switch):
    def boom(state):
        raise RuntimeError("display failed")

    switch, listener = make_switch(on_state_change=boom)
    listener.press(PAUSE)

    assert switch.state is SafetyState.PAUSED


# -------------------------------------------------------------------- lifecycle


def test_context_manager_starts_and_stops_listener(clock):
    listeners = []

    def factory(hotkeys):
        listener = FakeListener(hotkeys)
        listeners.append(listener)
        return listener

    with EmergencyKillSwitch(listener_factory=factory, clock=clock) as switch:
        assert listeners[0].started
        assert switch.state is SafetyState.RUNNING

    assert listeners[0].stopped


def test_stop_is_idempotent(make_switch):
    switch, listener = make_switch()
    switch.stop()
    switch.stop()

    assert listener.stopped


# ------------------------------------------------------------------ integration


@pytest.mark.integration
def test_real_listener_starts_and_stops():
    # Only checks that pynput can hook the keyboard here; sends no keystrokes.
    with EmergencyKillSwitch() as switch:
        assert switch.state is SafetyState.RUNNING
