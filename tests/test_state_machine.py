"""Tests for agent/state_machine.py and agent/recorder.py.

Everything around the loop is a fake: the screen returns the synthetic notes
app, the planner and the differ follow a script, and the controller and the
safety switch only record calls. The real VisualTagger draws the grids, so the
zoom regions and click points are the ones the agent would really use.
"""

import json

import pytest

from agent.prompts import StepRecord
from agent.recorder import RunRecorder
from agent.state_machine import (
    AgentStateMachine,
    RunStatus,
    StepEvent,
    estimate_cost,
)
from controller.safety import PausedError
from schemas.actions import ActionType, AgentAction
from tests import scenes
from vision.diff import DiffResult, Region

# ------------------------------------------------------------------- fakes


class FakeUsage:
    def __init__(
        self, input_tokens=1000, output_tokens=200, cache_read=0, cache_write=0
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read
        self.cache_creation_input_tokens = cache_write


class FakeScreen:
    def __init__(self):
        self.full = scenes.notes_app_frame()
        self.captures = 0
        self.regions = []

    def capture(self):
        self.captures += 1
        return self.full

    def capture_region(self, x, y, width, height):
        self.regions.append((x, y, width, height))
        return scenes.region_frame(x, y, width, height)


class FakePlanner:
    model = "claude-haiku-5-5"

    def __init__(self, actions, cells=()):
        self.actions = list(actions)
        self.cells = list(cells)
        self.decide_calls = []
        self.refine_calls = []
        self.usage = []

    def decide(self, goal, tagged, history=()):
        self.decide_calls.append(
            {"goal": goal, "tagged": tagged, "history": list(history)}
        )
        self.usage.append(FakeUsage())
        return self.actions.pop(0)

    def refine(self, goal, action, zoom_tagged, drag_end=False):
        self.refine_calls.append(
            {"action": action, "tagged": zoom_tagged, "drag_end": drag_end}
        )
        self.usage.append(FakeUsage())
        return self.cells.pop(0) if self.cells else None


class FakeController:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = dict(fail_on or {})  # call index -> exception to raise

    def _record(self, *call):
        index = len(self.calls)
        self.calls.append(call)
        if index in self.fail_on:
            raise self.fail_on.pop(index)

    def click(self, x, y, button="left"):
        self._record("click", x, y, button)

    def double_click(self, x, y):
        self._record("double_click", x, y)

    def type_text(self, text):
        self._record("type_text", text)

    def press_key(self, key):
        self._record("press_key", key)

    def drag(self, start_x, start_y, end_x, end_y, duration):
        self._record("drag", start_x, start_y, end_x, end_y, duration)

    def scroll(self, x, y, amount):
        self._record("scroll", x, y, amount)


class FakeSafety:
    def __init__(self, errors=()):
        self.errors = list(errors)  # one entry per check(): None or an exception
        self.checks = 0
        self.waits = 0
        self.abort_on_wait = False

    def check(self):
        self.checks += 1
        if self.errors:
            error = self.errors.pop(0)
            if error is not None:
                raise error

    def wait_while_paused(self, timeout=None):
        self.waits += 1
        if self.abort_on_wait:
            raise InterruptedError("aborted")
        return True


AREA = Region(x=1150, y=300, width=600, height=320)


class FakeDiffer:
    def __init__(self, changes=()):
        self.changes = list(changes)  # one bool per action; default True
        self.calls = 0

    def wait_for_change(self, before, capture):
        self.calls += 1
        changed = self.changes.pop(0) if self.changes else True
        if changed:
            result = DiffResult(True, 0.05, (AREA,), AREA)
        else:
            result = DiffResult(False, 0.0, (), None)
        return result, capture()


def act(kind, **fields):
    return AgentAction(thought=f"do {kind}", action_type=kind, **fields)


FINISH = act(ActionType.FINISH)


@pytest.fixture
def parts():
    return {
        "screen": FakeScreen(),
        "controller": FakeController(),
        "safety": FakeSafety(),
        "differ": FakeDiffer(),
        "sleeps": [],
    }


def run(parts, actions, cells=(), goal="Guardá el archivo", **kwargs):
    planner = FakePlanner(actions, cells)
    machine = AgentStateMachine(
        planner=planner,
        controller=parts["controller"],
        screen=parts["screen"],
        safety=parts["safety"],
        differ=parts["differ"],
        sleep=parts["sleeps"].append,
        **kwargs,
    )
    return machine.run(goal), planner


# ---------------------------------------------------------------- settings


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_steps": 0},
        {"stuck_limit": 0},
        {"max_wait": -1},
        {"zoom_grid": 1},
        {"zoom_margin": 0},
        {"zoom_margin": 3},
    ],
)
def test_invalid_settings_raise(parts, kwargs):
    with pytest.raises(ValueError):
        AgentStateMachine(
            planner=FakePlanner([]),
            controller=parts["controller"],
            screen=parts["screen"],
            safety=parts["safety"],
            **kwargs,
        )


# ------------------------------------------------------------- termination


def test_finish_right_away(parts):
    result, planner = run(parts, [FINISH])

    assert result.status is RunStatus.FINISHED
    assert result.steps == ()
    assert parts["controller"].calls == []
    assert planner.decide_calls[0]["goal"] == "Guardá el archivo"
    assert planner.decide_calls[0]["tagged"].grid_cols == 20


def test_fail_reports_the_models_reason(parts):
    result, _ = run(parts, [act(ActionType.FAIL)])

    assert result.status is RunStatus.FAILED
    assert result.reason == "do FAIL"


def test_stops_at_max_steps(parts):
    clicks = [act(ActionType.CLICK, target_id=24) for _ in range(5)]
    result, _ = run(parts, clicks, max_steps=3)

    assert result.status is RunStatus.MAX_STEPS
    assert len(result.steps) == 3
    assert len(parts["controller"].calls) == 3


def test_default_max_steps_is_10(parts):
    clicks = [act(ActionType.CLICK, target_id=24) for _ in range(12)]
    result, _ = run(parts, clicks)

    assert len(result.steps) == 10


def test_stuck_after_3_actions_without_visible_change(parts):
    parts["differ"] = FakeDiffer([False, False, False])
    keys = [act(ActionType.PRESS_KEY, key="tab") for _ in range(5)]
    result, _ = run(parts, keys)

    assert result.status is RunStatus.STUCK
    assert len(result.steps) == 3


def test_a_visible_change_resets_the_stuck_count(parts):
    parts["differ"] = FakeDiffer([False, False, True, False, False])
    keys = [act(ActionType.PRESS_KEY, key="tab") for _ in range(5)] + [FINISH]
    result, _ = run(parts, keys)

    assert result.status is RunStatus.FINISHED


def test_stuck_limit_is_configurable(parts):
    parts["differ"] = FakeDiffer([False])
    result, _ = run(parts, [act(ActionType.PRESS_KEY, key="tab")], stuck_limit=1)

    assert result.status is RunStatus.STUCK


def test_waits_do_not_count_as_stuck(parts):
    parts["differ"] = FakeDiffer([False] * 4)
    waits = [act(ActionType.WAIT, duration=0.1) for _ in range(4)] + [FINISH]
    result, _ = run(parts, waits)

    assert result.status is RunStatus.FINISHED


# --------------------------------------------------------- clicks and zoom


def test_click_zooms_on_the_cell_and_half_its_neighbours(parts):
    result, planner = run(
        parts, [act(ActionType.CLICK, target_id=24), FINISH], cells=[19]
    )

    # Cell 24 is 96x90 screen px; half a cell around it -> 192x180 px.
    assert parts["screen"].regions == [(336, 45, 192, 180)]
    (refine,) = planner.refine_calls
    assert (refine["tagged"].grid_cols, refine["tagged"].grid_rows) == (8, 8)
    assert refine["drag_end"] is False
    assert parts["controller"].calls == [("click", 420, 101, "left")]
    assert scenes.inside_save_button(420, 101)


def test_zoom_region_is_clipped_at_the_screen_corner(parts):
    run(parts, [act(ActionType.CLICK, target_id=0), FINISH], cells=[0])

    assert parts["screen"].regions == [(0, 0, 144, 135)]


def test_zoom_margin_of_one_cell(parts):
    run(
        parts,
        [act(ActionType.CLICK, target_id=24), FINISH],
        cells=[19],
        zoom_margin=1.0,
    )

    assert parts["screen"].regions == [(288, 0, 288, 270)]
    assert parts["controller"].calls == [("click", 414, 84, "left")]


def test_zoom_grid_is_configurable(parts):
    _, planner = run(
        parts, [act(ActionType.CLICK, target_id=24), FINISH], cells=[5], zoom_grid=6
    )

    assert planner.refine_calls[0]["tagged"].grid_cols == 6


def test_falls_back_to_the_coarse_cell_center_when_zoom_fails(parts):
    run(parts, [act(ActionType.CLICK, target_id=24), FINISH], cells=[])

    assert parts["controller"].calls == [("click", 432, 135, "left")]


def test_double_and_right_click(parts):
    actions = [
        act(ActionType.DOUBLE_CLICK, target_id=24),
        act(ActionType.RIGHT_CLICK, target_id=24),
        FINISH,
    ]
    run(parts, actions, cells=[19, 19])

    assert parts["controller"].calls == [
        ("double_click", 420, 101),
        ("click", 420, 101, "right"),
    ]


def test_scroll_zooms_then_scrolls(parts):
    run(
        parts,
        [act(ActionType.SCROLL, target_id=24, scroll_amount=3), FINISH],
        cells=[19],
    )

    assert parts["controller"].calls == [("scroll", 420, 101, 3)]


def test_drag_refines_both_ends(parts):
    drag = act(ActionType.DRAG, target_id=24, drag_end_target_id=24, duration=0.8)
    _, planner = run(parts, [drag, FINISH], cells=[19, 20])

    assert [c["drag_end"] for c in planner.refine_calls] == [False, True]
    (call,) = parts["controller"].calls
    assert call[0] == "drag" and call[1:3] == (420, 101) and call[5] == 0.8
    assert call[3] > call[1]  # cell 20 is right of cell 19


# ----------------------------------------------------------- keyboard, wait


def test_type_and_press_key_skip_the_zoom(parts):
    actions = [
        act(ActionType.TYPE, text="Hola Mundo"),
        act(ActionType.PRESS_KEY, key="ctrl+s"),
        FINISH,
    ]
    _, planner = run(parts, actions)

    assert planner.refine_calls == []
    assert parts["screen"].regions == []
    assert parts["controller"].calls == [
        ("type_text", "Hola Mundo"),
        ("press_key", "ctrl+s"),
    ]


def test_wait_sleeps_in_short_slices_checking_safety(parts):
    run(parts, [act(ActionType.WAIT, duration=1.0), FINISH])

    assert sum(parts["sleeps"]) == pytest.approx(1.0)
    assert max(parts["sleeps"]) <= 0.1 + 1e-9
    assert parts["safety"].checks >= 10


def test_wait_is_capped_at_10_seconds(parts):
    run(parts, [act(ActionType.WAIT, duration=30.0), FINISH])

    assert sum(parts["sleeps"]) == pytest.approx(10.0)


def test_wait_cap_is_configurable(parts):
    run(parts, [act(ActionType.WAIT, duration=30.0), FINISH], max_wait=2.0)

    assert sum(parts["sleeps"]) == pytest.approx(2.0)


# -------------------------------------------------- perception and history


def test_history_carries_what_changed(parts):
    parts["differ"] = FakeDiffer([True, False])
    actions = [
        act(ActionType.CLICK, target_id=24),
        act(ActionType.PRESS_KEY, key="tab"),
        FINISH,
    ]
    result, planner = run(parts, actions, cells=[19])

    history = planner.decide_calls[2]["history"]
    assert history == list(result.steps)
    assert history[0] == StepRecord(
        actions[0], screen_changed=True, changed_area=(1150, 300, 600, 320)
    )
    assert history[1].screen_changed is False
    assert history[1].changed_area is None


def test_frame_after_an_action_is_reused_as_the_next_perception(parts):
    actions = [act(ActionType.PRESS_KEY, key="tab") for _ in range(3)] + [FINISH]
    run(parts, actions)

    # One capture to start, then the differ's last frame each step.
    assert parts["screen"].captures == 1 + 3


# ------------------------------------------------------------ pause / abort


def test_pause_during_an_action_discards_it_and_perceives_again(parts):
    parts["controller"] = FakeController(fail_on={0: PausedError("paused")})
    actions = [
        act(ActionType.CLICK, target_id=24),
        act(ActionType.CLICK, target_id=24),
        FINISH,
    ]
    result, planner = run(parts, actions, cells=[19, 19])

    assert parts["safety"].waits == 1
    assert len(planner.decide_calls) == 3
    assert len(result.steps) == 1  # the interrupted click is not in the history
    assert result.status is RunStatus.FINISHED
    assert parts["screen"].captures == 1 + 1 + 1  # start, after pause, after click


def test_pause_while_the_model_thinks_skips_the_action(parts):
    # check() order: loop start (ok), before acting (paused).
    parts["safety"] = FakeSafety([None, PausedError("paused")])
    actions = [act(ActionType.TYPE, text="no"), FINISH]
    result, _ = run(parts, actions)

    assert parts["controller"].calls == []
    assert parts["safety"].waits == 1
    assert result.status is RunStatus.FINISHED


def test_abort_during_an_action(parts):
    parts["controller"] = FakeController(fail_on={0: InterruptedError("stop")})
    result, _ = run(parts, [act(ActionType.TYPE, text="x"), FINISH])

    assert result.status is RunStatus.ABORTED


def test_abort_while_paused(parts):
    parts["controller"] = FakeController(fail_on={0: PausedError("paused")})
    parts["safety"].abort_on_wait = True
    result, _ = run(parts, [act(ActionType.TYPE, text="x"), FINISH])

    assert result.status is RunStatus.ABORTED


def test_abort_before_the_first_step(parts):
    parts["safety"] = FakeSafety([InterruptedError("stop")])
    result, planner = run(parts, [FINISH])

    assert result.status is RunStatus.ABORTED
    assert planner.decide_calls == []


# ------------------------------------------------------- progress and cost


def test_on_step_reports_each_executed_step(parts):
    events = []
    actions = [
        act(ActionType.CLICK, target_id=24),
        act(ActionType.TYPE, text="hola"),
        FINISH,
    ]
    run(parts, actions, cells=[19], on_step=events.append)

    assert [e.number for e in events] == [1, 2]
    assert all(isinstance(e, StepEvent) for e in events)
    assert events[0].point == (420, 101)
    assert events[0].screen_changed is True
    assert events[1].point is None
    # Haiku: 1000 in + 200 out = US$ 0.0002 per call; 3 calls before step 2 ends.
    assert events[1].cost_usd == pytest.approx(0.0006)


def test_result_carries_the_total_cost(parts):
    result, _ = run(parts, [act(ActionType.CLICK, target_id=24), FINISH], cells=[19])

    assert result.cost_usd == pytest.approx(3 * 0.0002)


def test_estimate_cost_for_haiku_with_cache():
    usage = FakeUsage(input_tokens=1279, output_tokens=179, cache_read=2177)

    # The first dry run: 1279 x 0.10 + 2177 x 0.01 + 179 x 0.50, per million.
    assert estimate_cost([usage], "claude-haiku-5-5") == pytest.approx(
        0.00024, abs=1e-5
    )


def test_estimate_cost_unknown_model_is_zero():
    assert estimate_cost([FakeUsage()], "some-other-model") == 0.0


# --------------------------------------------------------------- recorder


def test_recorder_writes_one_folder_per_run(parts, tmp_path):
    recorder = RunRecorder(base_dir=tmp_path, keep=10)
    actions = [
        act(ActionType.CLICK, target_id=24),
        act(ActionType.TYPE, text="hola"),
        FINISH,
    ]
    result, _ = run(parts, actions, cells=[19], recorder=recorder)

    run_dir = result.run_dir
    assert run_dir is not None and run_dir.parent == tmp_path
    assert (run_dir / "step_01.png").exists()
    assert (run_dir / "step_02.png").exists()
    lines = (run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    assert first["action"]["action_type"] == "CLICK"
    assert first["point"] == [420, 101]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "finished"
    assert summary["goal"] == "Guardá el archivo"
    assert summary["steps"] == 2


def test_recorder_keeps_only_the_last_runs(tmp_path):
    for i in range(12):
        (tmp_path / f"20260101-0000{i:02d}").mkdir()
    recorder = RunRecorder(base_dir=tmp_path, keep=10)
    new_dir = recorder.start("goal")

    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert len(remaining) == 10
    assert new_dir.name in remaining
    assert "20260101-000000" not in remaining  # oldest removed


def test_recorder_default_keeps_10():
    assert RunRecorder().keep == 10


def test_recorder_never_deletes_unrelated_folders(tmp_path):
    (tmp_path / "notes").mkdir()
    for i in range(12):
        (tmp_path / f"20260101-0000{i:02d}").mkdir()
    RunRecorder(base_dir=tmp_path, keep=2).start("goal")

    assert (tmp_path / "notes").exists()
