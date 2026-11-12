"""Tests for agent/recorder.py: per-step timings in steps.jsonl and summary.json."""

import json

import numpy as np
import pytest

from agent.recorder import RunRecorder
from agent.state_machine import RunResult, RunStatus, StepEvent
from schemas.actions import ActionType, AgentAction


def timings(**overrides):
    base = dict.fromkeys(
        ("capture", "tag", "decide", "refine", "act", "wait", "record", "total"), 0.0
    )
    base.update(overrides)
    return base


def event(number, stage_timings):
    return StepEvent(
        number=number,
        action=AgentAction(thought="escribo", action_type=ActionType.TYPE, text="hola"),
        point=None,
        end_point=None,
        screen_changed=True,
        changed_area=None,
        cost_usd=0.001,
        timings=stage_timings,
    )


def result():
    return RunResult(
        status=RunStatus.FINISHED,
        reason="listo",
        steps=(),
        cost_usd=0.002,
    )


IMAGE = np.zeros((4, 4, 3), dtype=np.uint8)


def read_lines(run_dir):
    text = (run_dir / "steps.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def test_each_step_line_carries_its_timings(tmp_path):
    recorder = RunRecorder(base_dir=tmp_path)
    run_dir = recorder.start("goal")
    first = timings(decide=1.5, total=2.0)
    recorder.record_step(event(1, first), IMAGE)

    (line,) = read_lines(run_dir)
    assert line["timings"] == first


def test_summary_has_per_stage_totals_and_averages(tmp_path):
    recorder = RunRecorder(base_dir=tmp_path)
    run_dir = recorder.start("goal")
    recorder.record_step(event(1, timings(decide=1.0, wait=0.5, total=2.0)), IMAGE)
    recorder.record_step(event(2, timings(decide=3.0, wait=0.0, total=4.0)), IMAGE)
    recorder.finish(result())

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["timings"]["totals"]["decide"] == pytest.approx(4.0)
    assert summary["timings"]["totals"]["wait"] == pytest.approx(0.5)
    assert summary["timings"]["totals"]["total"] == pytest.approx(6.0)
    assert summary["timings"]["averages"]["decide"] == pytest.approx(2.0)
    assert summary["timings"]["averages"]["total"] == pytest.approx(3.0)


def test_summary_of_a_run_without_steps_has_zero_averages(tmp_path):
    recorder = RunRecorder(base_dir=tmp_path)
    run_dir = recorder.start("goal")
    recorder.finish(result())

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["timings"] == {"totals": {}, "averages": {}}


def test_timings_of_a_new_run_do_not_leak_into_the_next(tmp_path):
    recorder = RunRecorder(base_dir=tmp_path)
    recorder.start("first")
    recorder.record_step(event(1, timings(decide=9.0)), IMAGE)
    recorder.finish(result())

    run_dir = recorder.start("second")
    recorder.finish(result())

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert "decide" not in summary["timings"]["totals"]
