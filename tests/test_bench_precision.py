"""Tests for bench/precision.py and tests/bench_scenes.py (no API calls).

A fake planner stands in for Claude: it answers with the grid cell nearest to
the scene's target (or scripted wrong ones), so scoring, the budget stop and
the report format can be checked deterministically.
"""

import json

import numpy as np
import pytest

from agent.state_machine import estimate_cost
from bench import precision
from bench.precision import (
    ArrayScreen,
    SceneResult,
    run_bench,
    run_scene,
    save_results,
    score_point,
    summarize,
)
from schemas.actions import ActionType, AgentAction
from tests import bench_scenes

MODEL = "claude-sonnet-5-5"


class FakeUsage:
    input_tokens = 2000
    output_tokens = 100


class FakePlanner:
    """Picks the grid cell nearest to the scene's target center."""

    def __init__(self, scene_by_goal, offset=(0, 0), refine_result="best"):
        self.model = MODEL
        self.usage: list = []
        self._scenes = scene_by_goal
        self._offset = offset
        self._refine_result = refine_result

    @staticmethod
    def _nearest(tagged, point):
        best, best_d = 0, float("inf")
        for cell in range(tagged.cell_count):
            cx, cy = tagged.cell_to_screen(cell)
            d = (cx - point[0]) ** 2 + (cy - point[1]) ** 2
            if d < best_d:
                best, best_d = cell, d
        return best

    def decide(self, goal, tagged, history=()):
        self.usage.append(FakeUsage())
        cell = self._nearest(tagged, self._scenes[goal].center)
        return AgentAction(thought="t", action_type=ActionType.CLICK, target_id=cell)

    def refine(self, goal, action, zoom_tagged, drag_end=False):
        self.usage.append(FakeUsage())
        if self._refine_result is None:
            return None
        cx, cy = self._scenes[goal].center
        return self._nearest(zoom_tagged, (cx + self._offset[0], cy + self._offset[1]))


@pytest.fixture(scope="module")
def scenes():
    return bench_scenes.all_scenes()


@pytest.fixture
def by_goal(scenes):
    return {s.goal: s for s in scenes}


@pytest.fixture
def save_scene(scenes):
    return next(s for s in scenes if s.name == "notes_save_button")


# ------------------------------------------------------------------ scenes


def test_there_are_about_twenty_varied_scenes(scenes):
    assert len(scenes) >= 20
    assert len({s.name for s in scenes}) == len(scenes)
    assert len({s.goal for s in scenes}) == len(scenes)
    assert {"large", "small", "edge"} <= {s.kind for s in scenes}
    assert sum(s.kind == "edge" for s in scenes) >= 5


def test_scenes_are_full_hd_with_the_target_on_screen(scenes):
    for s in scenes:
        h, w = s.image.shape[:2]
        assert (w, h) == (1920, 1080), s.name
        x, y, tw, th = s.target
        assert x >= 0 and y >= 0 and x + tw <= w and y + th <= h, s.name
        assert tw > 0 and th > 0, s.name


def test_scene_geometry_helpers():
    scene = bench_scenes.Scene(
        "n", "large", "g", np.zeros((10, 10, 3), np.uint8), (10, 20, 40, 10)
    )
    assert scene.center == (30, 25)
    assert scene.contains(10, 20)
    assert scene.contains(49, 29)
    assert not scene.contains(50, 25)
    assert not scene.contains(30, 30)


def test_scenes_are_deterministic():
    a, b = bench_scenes.all_scenes(), bench_scenes.all_scenes()
    assert all(np.array_equal(x.image, y.image) for x, y in zip(a, b, strict=True))


# ----------------------------------------------------------------- scoring


def test_score_point_inside_and_distance():
    assert score_point((30, 25), (10, 20, 40, 10)) == (True, 0.0)
    hit, dist = score_point((30, 45), (10, 20, 40, 10))
    assert not hit and dist == 20
    hit, dist = score_point((33, 29), (10, 20, 40, 10))
    assert hit and dist == pytest.approx(5.0)


def test_score_point_without_a_point_is_a_miss():
    assert score_point(None, (0, 0, 10, 10)) == (False, None)


# ------------------------------------------------------------------- screen


def test_array_screen_mimics_screen_capture():
    img = np.zeros((1080, 1920, 3), np.uint8)
    img[100:200, 300:400] = 255
    screen = ArrayScreen(img)
    full = screen.capture()
    assert full.size == (1280, 720)
    assert full.original_size == (1920, 1080)
    assert full.scale == pytest.approx(1280 / 1920)
    region = screen.capture_region(300, 100, 100, 100)
    assert region.scale == 1.0 and region.origin == (300, 100)
    assert region.image.min() == 255
    with pytest.raises(ValueError):
        screen.capture_region(1900, 1000, 100, 100)


# --------------------------------------------------------------- run_scene


def test_run_scene_hits_with_a_good_planner(save_scene, by_goal):
    planner = FakePlanner(by_goal)
    result = run_scene(save_scene, planner)
    assert isinstance(result, SceneResult)
    assert result.hit
    assert result.calls == 2
    assert result.distance_px < 25
    assert result.cost_usd == pytest.approx(estimate_cost(planner.usage, MODEL))
    assert result.action_type == "click"
    assert result.error is None


def test_run_scene_misses_when_refine_lands_outside(save_scene, by_goal):
    result = run_scene(save_scene, FakePlanner(by_goal, offset=(0, 200)))
    assert not result.hit
    assert result.distance_px > 100


def test_run_scene_falls_back_to_the_coarse_cell(save_scene, by_goal):
    result = run_scene(save_scene, FakePlanner(by_goal, refine_result=None))
    assert result.point == result.coarse_point


def test_run_scene_non_click_action_is_a_miss(scenes):
    class Finisher(FakePlanner):
        def decide(self, goal, tagged, history=()):
            self.usage.append(FakeUsage())
            return AgentAction(thought="done", action_type=ActionType.FINISH)

    result = run_scene(scenes[0], Finisher({}))
    assert not result.hit
    assert result.point is None and result.distance_px is None
    assert result.calls == 1
    assert result.action_type == "finish"


def test_run_scene_records_errors_instead_of_raising(scenes):
    class Broken(FakePlanner):
        def decide(self, goal, tagged, history=()):
            raise RuntimeError("boom")

    result = run_scene(scenes[0], Broken({}))
    assert not result.hit
    assert "boom" in result.error


# -------------------------------------------------------------- run_bench


def test_run_bench_runs_the_given_scenes(scenes, by_goal):
    results, stopped = run_bench(scenes[:3], FakePlanner(by_goal), budget=10.0)
    assert [r.name for r in results] == [s.name for s in scenes[:3]]
    assert not stopped


def test_run_bench_stops_when_the_budget_is_reached(scenes, by_goal):
    one_scene = 2 * estimate_cost([FakeUsage()], MODEL)
    results, stopped = run_bench(scenes, FakePlanner(by_goal), budget=one_scene * 2.5)
    assert stopped
    assert len(results) == 3  # the check happens before each scene


def test_run_bench_zero_budget_runs_nothing(scenes, by_goal):
    results, stopped = run_bench(scenes, FakePlanner(by_goal), budget=0.0)
    assert results == [] and stopped


# ------------------------------------------------------------------ report


def make_results():
    base = {
        "kind": "large",
        "goal": "g",
        "target": (0, 0, 10, 10),
        "action_type": "click",
        "coarse_cell": 1,
        "coarse_point": (1, 1),
        "error": None,
    }
    return [
        SceneResult(
            name="a", point=(5, 5), hit=True, distance_px=0.0,
            calls=2, cost_usd=0.01, **base,
        ),
        SceneResult(
            name="b", point=(5, 25), hit=False, distance_px=20.0,
            calls=2, cost_usd=0.01, **base,
        ),
        SceneResult(
            name="c", point=None, hit=False, distance_px=None,
            calls=1, cost_usd=0.005, **base,
        ),
    ]  # fmt: skip


def test_summarize():
    s = summarize(make_results(), MODEL, "low", stopped_by_budget=False)
    assert s["scenes"] == 3
    assert s["hits"] == 1
    assert s["hit_rate"] == pytest.approx(1 / 3)
    assert s["mean_distance_px"] == pytest.approx(10.0)
    assert s["median_distance_px"] == pytest.approx(10.0)
    assert s["calls"] == 5
    assert s["cost_usd"] == pytest.approx(0.025)
    assert s["no_point"] == 1
    assert s["by_kind"]["large"]["scenes"] == 3
    assert s["model"] == MODEL and s["effort"] == "low"
    assert s["stopped_by_budget"] is False


def test_summarize_empty():
    s = summarize([], MODEL, "low", stopped_by_budget=True)
    assert s["scenes"] == 0 and s["hit_rate"] is None
    assert s["mean_distance_px"] is None


def test_save_results_writes_rows_and_summary(tmp_path):
    results = make_results()
    summary = summarize(results, MODEL, "low", stopped_by_budget=False)
    path = save_results(results, summary, tmp_path / "out" / "r.json")
    data = json.loads(path.read_text())
    assert set(data) == {"summary", "scenes"}
    assert len(data["scenes"]) == 3
    assert data["scenes"][0]["name"] == "a"
    assert data["scenes"][0]["target"] == [0, 0, 10, 10]
    assert data["summary"]["hits"] == 1


def test_default_result_path_has_timestamp_and_model(tmp_path):
    path = precision.default_result_path("claude-sonnet-5-5", root=tmp_path)
    assert path.parent == tmp_path
    assert path.name.endswith("-claude-sonnet-5-5.json")


def test_format_report_mentions_the_key_numbers():
    summary = summarize(make_results(), MODEL, "low", stopped_by_budget=False)
    text = precision.format_report(summary).lower()
    assert "33" in text and "hit rate" in text
    assert "mean distance" in text


def test_main_requires_a_key_and_never_prints_it(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    empty = tmp_path / ".env"
    empty.write_text("")
    code = precision.main(["--env-file", str(empty), "--scenes", "1"])
    assert code != 0
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
