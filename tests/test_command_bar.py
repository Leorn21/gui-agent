"""Tests for ui/command_bar.py: everything except the tkinter window itself.

The window is a thin layer over these pieces and is checked by hand: the
status pill's geometry, the texts it shows, the screen mask that hides the
pill from the model, the controller guard that keeps clicks off it, and the
runner that executes the agent in a background thread.
"""

import threading

import numpy as np
import pytest

from agent.state_machine import RunResult, RunStatus, StepEvent
from schemas.actions import ActionType, AgentAction
from tests import scenes
from ui.command_bar import (
    MASK_COLOR,
    PILL_MARGIN,
    PILL_SIZE,
    AgentRunner,
    GuardedController,
    MaskedScreen,
    pill_rect,
    result_text,
    step_text,
)

WORK_AREA = (0, 0, 1920, 1032)  # primary monitor minus a 48 px taskbar
PILL = pill_rect(WORK_AREA)


def event(kind=ActionType.CLICK, number=3, **fields):
    if kind is ActionType.CLICK:
        fields.setdefault("target_id", 24)
    action = AgentAction(thought="t", action_type=kind, **fields)
    return StepEvent(
        number=number,
        action=action,
        point=(414, 84),
        end_point=None,
        screen_changed=True,
        changed_area=None,
        cost_usd=0.0012,
    )


def result(status):
    return RunResult(status=status, reason="why", steps=(), cost_usd=0.0016)


# ------------------------------------------------------------------ geometry


def test_pill_sits_bottom_right_inside_the_work_area():
    w, h = PILL_SIZE
    assert PILL == (1920 - PILL_MARGIN - w, 1032 - PILL_MARGIN - h, w, h)


def test_pill_respects_a_work_area_offset():
    x, y, w, h = pill_rect((-1280, 0, 1280, 984))

    assert x + w == -PILL_MARGIN
    assert y + h == 984 - PILL_MARGIN


# --------------------------------------------------------------------- texts


def test_step_text_shows_step_action_and_pause_hint():
    text = step_text(event(number=3))

    assert "Paso 3" in text
    assert "CLICK" in text
    assert "Ctrl+Alt+Q" in text


def test_step_text_uses_the_configured_hotkey():
    assert "Ctrl+Shift+P" in step_text(event(), pause_hotkey="ctrl+shift+p")


@pytest.mark.parametrize(
    ("status", "word"),
    [
        (RunStatus.FINISHED, "Listo"),
        (RunStatus.FAILED, "No pudo completarse"),
        (RunStatus.STUCK, "Se atascó"),
        (RunStatus.ABORTED, "Detenido"),
        (RunStatus.MAX_STEPS, "Límite de pasos"),
    ],
)
def test_result_text_is_only_the_status(status, word):
    text = result_text(result(status))

    assert text == word
    assert "US$" not in text and "why" not in text


# ---------------------------------------------------------------- the mask


class Screen:
    def __init__(self):
        self.full = scenes.notes_app_frame()
        self.region_calls = []

    def capture(self):
        return self.full

    def capture_region(self, x, y, w, h):
        self.region_calls.append((x, y, w, h))
        return scenes.region_frame(x, y, w, h)


def test_capture_paints_the_pill_area_in_image_coordinates():
    base = Screen()
    frame = MaskedScreen(base, PILL).capture()

    px, py, pw, ph = PILL
    s = frame.scale
    inside = frame.image[int((py + ph / 2) * s), int((px + pw / 2) * s)]
    assert tuple(int(v) for v in inside) == MASK_COLOR
    assert np.array_equal(frame.image[:400, :800], base.full.image[:400, :800])
    assert (frame.scale, frame.origin, frame.original_size) == (
        base.full.scale,
        base.full.origin,
        base.full.original_size,
    )


def test_capture_does_not_modify_the_underlying_frame():
    base = Screen()
    before = base.full.image.copy()
    MaskedScreen(base, PILL).capture()

    assert np.array_equal(base.full.image, before)


def test_region_overlapping_the_pill_is_painted():
    px, py, _, _ = PILL
    region = MaskedScreen(Screen(), PILL).capture_region(px - 50, py - 50, 100, 100)

    assert tuple(int(v) for v in region.image[75, 75]) == MASK_COLOR  # inside
    assert tuple(int(v) for v in region.image[10, 10]) != MASK_COLOR  # outside


def test_region_far_from_the_pill_is_untouched():
    base = Screen()
    region = MaskedScreen(base, PILL).capture_region(288, 0, 288, 270)

    assert np.array_equal(region.image, scenes.region_frame(288, 0, 288, 270).image)
    assert base.region_calls == [(288, 0, 288, 270)]


# -------------------------------------------------------------- the guard


class Recorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.calls.append((name, args, kwargs))


INSIDE = (PILL[0] + 10, PILL[1] + 10)
OUTSIDE = (414, 84)


def test_actions_outside_the_pill_are_forwarded():
    inner = Recorder()
    guard = GuardedController(inner, PILL)
    guard.click(*OUTSIDE)
    guard.click(*OUTSIDE, button="right")
    guard.double_click(*OUTSIDE)
    guard.scroll(*OUTSIDE, 3)
    guard.drag(*OUTSIDE, 600, 400, 0.5)
    guard.type_text("hola")
    guard.press_key("enter")

    assert [c[0] for c in inner.calls] == [
        "click",
        "click",
        "double_click",
        "scroll",
        "drag",
        "type_text",
        "press_key",
    ]


@pytest.mark.parametrize(
    "call",
    [
        lambda g: g.click(*INSIDE),
        lambda g: g.double_click(*INSIDE),
        lambda g: g.scroll(*INSIDE, 2),
        lambda g: g.drag(*OUTSIDE, *INSIDE, 0.5),
        lambda g: g.drag(*INSIDE, *OUTSIDE, 0.5),
    ],
)
def test_actions_on_the_pill_are_skipped_and_reported(call):
    inner, blocked = Recorder(), []
    call(GuardedController(inner, PILL, on_blocked=blocked.append))

    assert inner.calls == []
    assert len(blocked) == 1


# --------------------------------------------------------------- the runner


class FakeMachine:
    def __init__(self, on_step, gate=None, error=None):
        self.on_step = on_step
        self.gate = gate
        self.error = error
        self.goals = []

    def run(self, goal):
        self.goals.append(goal)
        if self.gate:
            self.gate.wait(2)
        if self.error:
            raise self.error
        self.on_step(event(number=1))
        return result(RunStatus.FINISHED)


def make_runner(machine_kwargs=None, **kwargs):
    built, events, done, sleeps = [], [], [], []

    def build(on_step):
        machine = FakeMachine(on_step, **(machine_kwargs or {}))
        built.append(machine)
        return machine

    runner = AgentRunner(
        build,
        on_event=events.append,
        on_done=lambda res, err: done.append((res, err)),
        sleep=sleeps.append,
        **kwargs,
    )
    return runner, built, events, done, sleeps


def test_runs_the_goal_in_the_background_after_a_silent_delay():
    runner, built, events, done, sleeps = make_runner()

    assert runner.start("Abrí el Bloc de notas") is True
    runner.join(2)

    assert sleeps == [1.0]
    assert built[0].goals == ["Abrí el Bloc de notas"]
    assert [e.number for e in events] == [1]
    assert done[0][0].status is RunStatus.FINISHED and done[0][1] is None
    assert runner.is_running is False


def test_start_delay_is_configurable():
    runner, *_, sleeps = make_runner(start_delay=0.0)
    runner.start("x")
    runner.join(2)

    assert sleeps == [0.0]


@pytest.mark.parametrize("goal", ["", "   "])
def test_empty_goals_are_ignored(goal):
    runner, built, *_ = make_runner()

    assert runner.start(goal) is False
    assert built == []


def test_cannot_start_while_running():
    gate = threading.Event()
    runner, built, *_ = make_runner({"gate": gate})
    runner.start("first")

    assert runner.is_running is True
    assert runner.start("second") is False
    gate.set()
    runner.join(2)
    assert len(built) == 1


def test_errors_are_reported_not_raised():
    boom = RuntimeError("API down")
    runner, _, _, done, _ = make_runner({"error": boom})
    runner.start("x")
    runner.join(2)

    assert done == [(None, boom)]
    assert runner.is_running is False
