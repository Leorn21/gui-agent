"""Precision bench: how close do the agent's clicks land to the target?

For each synthetic scene (``tests/bench_scenes.py``) it runs the real
pipeline pieces without touching the mouse or the screen: tag the frame,
``Planner.decide``, the zoom (``AgentStateMachine._locate``, so the region,
the margin and the fine grid are exactly the agent's) and ``Planner.refine``.
The resulting click point is scored against the target rectangle.

    python -m bench.precision --scenes 20 --budget 1.0
    python -m bench.precision --baseline        # also refresh bench/baseline.json

The run stops before a scene once the estimated cost reaches ``--budget``.
The API key is read from the environment or from ``--env-file`` and is never
printed or stored.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from agent.state_machine import AgentStateMachine, estimate_cost
from schemas.actions import ActionType
from tests.bench_scenes import Rect, Scene, all_scenes
from vision.capture import DEFAULT_MAX_WIDTH, Frame
from vision.som_tagger import VisualTagger

RESULTS_DIR = Path(__file__).parent / "results"
BASELINE_PATH = Path(__file__).parent / "baseline.json"
DEFAULT_BUDGET = 1.0
_CLICKS = (ActionType.CLICK, ActionType.DOUBLE_CLICK, ActionType.RIGHT_CLICK)


class ArrayScreen:
    """A ``Screen`` backed by an image, as ``ScreenCapture`` would return it.

    ``capture`` is downscaled to ``max_width``; ``capture_region`` is cropped
    at full resolution (and only downscaled when wider than ``max_width``).
    """

    def __init__(
        self, image: NDArray[np.uint8], max_width: int = DEFAULT_MAX_WIDTH
    ) -> None:
        self._image = image
        self._max_width = max_width

    def _frame(self, x: int, y: int, width: int, height: int) -> Frame:
        image = self._image[y : y + height, x : x + width]
        scale = 1.0
        if width > self._max_width:
            scale = self._max_width / width
            image = cv2.resize(
                image,
                (self._max_width, max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return Frame(
            image=np.ascontiguousarray(image),
            original_size=(width, height),
            scale=scale,
            origin=(x, y),
        )

    def capture(self) -> Frame:
        height, width = self._image.shape[:2]
        return self._frame(0, 0, width, height)

    def capture_region(self, x: int, y: int, width: int, height: int) -> Frame:
        full_h, full_w = self._image.shape[:2]
        if (
            width <= 0
            or height <= 0
            or x < 0
            or y < 0
            or x + width > full_w
            or y + height > full_h
        ):
            raise ValueError(
                f"Region ({x}, {y}, {width}x{height}) is outside the image"
            )
        return self._frame(x, y, width, height)


@dataclass(frozen=True)
class SceneResult:
    """Outcome of one scene.

    ``point`` is the final click point (``None`` when the planner did not
    choose a click); ``coarse_point`` is where the click would have landed
    without the zoom (center of the coarse cell).
    """

    name: str
    kind: str
    goal: str
    target: Rect
    action_type: str
    coarse_cell: int | None
    coarse_point: tuple[int, int] | None
    point: tuple[int, int] | None
    hit: bool
    distance_px: float | None
    calls: int
    cost_usd: float
    error: str | None = None


def score_point(
    point: tuple[float, float] | None, target: Rect
) -> tuple[bool, float | None]:
    """``(hit, distance to the target center in px)``; a missing point is a miss."""

    if point is None:
        return False, None
    x, y, w, h = target
    hit = x <= point[0] < x + w and y <= point[1] < y + h
    return hit, math.hypot(point[0] - (x + w // 2), point[1] - (y + h // 2))


def run_scene(scene: Scene, planner: Any) -> SceneResult:
    """Run the agent's click pipeline on one scene. Errors are recorded, not raised."""

    screen = ArrayScreen(scene.image)
    # Never moves the mouse: only the zoom helpers of the state machine are used.
    machine = AgentStateMachine(planner, controller=None, screen=screen, safety=None)  # type: ignore[arg-type]
    calls_before = len(planner.usage)
    action_type, coarse_cell, coarse_point, point = "none", None, None, None
    error = None
    try:
        tagged = VisualTagger().apply_labeled_grid(screen.capture())
        action = planner.decide(scene.goal, tagged, [])
        action_type = action.action_type.value
        if action.action_type in _CLICKS:
            coarse_cell = action.target_id
            if coarse_cell is not None:
                coarse_point = tagged.cell_to_screen(coarse_cell)
            point, _ = machine._locate(scene.goal, action, tagged)
    except Exception as exc:  # a failed call must not abort the whole bench
        error = f"{type(exc).__name__}: {exc}"

    hit, distance = score_point(point, scene.target)
    new_usage = planner.usage[calls_before:]
    return SceneResult(
        name=scene.name,
        kind=scene.kind,
        goal=scene.goal,
        target=scene.target,
        action_type=action_type,
        coarse_cell=coarse_cell,
        coarse_point=coarse_point,
        point=point,
        hit=hit,
        distance_px=distance,
        calls=len(new_usage),
        cost_usd=estimate_cost(new_usage, planner.model),
        error=error,
    )


def run_bench(
    scenes: Sequence[Scene], planner: Any, budget: float = DEFAULT_BUDGET
) -> tuple[list[SceneResult], bool]:
    """Run ``scenes`` in order, stopping once the estimated cost reaches ``budget``.

    The budget is checked before each scene, so the last one may overshoot it
    by at most the cost of one scene.

    Returns:
        ``(results, stopped_by_budget)``.
    """

    results: list[SceneResult] = []
    for scene in scenes:
        if estimate_cost(planner.usage, planner.model) >= budget:
            return results, True
        results.append(run_scene(scene, planner))
    return results, False


# ------------------------------------------------------------------- report


def _stats(rows: Sequence[SceneResult]) -> dict[str, Any]:
    distances = [r.distance_px for r in rows if r.distance_px is not None]
    hits = sum(r.hit for r in rows)
    return {
        "scenes": len(rows),
        "hits": hits,
        "hit_rate": hits / len(rows) if rows else None,
        "mean_distance_px": statistics.fmean(distances) if distances else None,
        "median_distance_px": statistics.median(distances) if distances else None,
        "coarse_hits": sum(
            r.coarse_point is not None and score_point(r.coarse_point, r.target)[0]
            for r in rows
        ),
    }


def summarize(
    results: Sequence[SceneResult],
    model: str,
    effort: str,
    stopped_by_budget: bool,
) -> dict[str, Any]:
    """Aggregate numbers of a run, overall and per scene kind."""

    summary = _stats(results)
    summary.update(
        {
            "no_point": sum(r.point is None for r in results),
            "calls": sum(r.calls for r in results),
            "cost_usd": sum(r.cost_usd for r in results),
            "by_kind": {
                kind: _stats([r for r in results if r.kind == kind])
                for kind in sorted({r.kind for r in results})
            },
            "model": model,
            "effort": effort,
            "stopped_by_budget": stopped_by_budget,
        }
    )
    return summary


def default_result_path(model: str, root: Path = RESULTS_DIR) -> Path:
    """``<root>/<timestamp>-<model>.json``."""

    return root / f"{datetime.now():%Y%m%d-%H%M%S}-{model}.json"


def save_results(
    results: Sequence[SceneResult], summary: dict[str, Any], path: Path
) -> Path:
    """Write ``{"summary": ..., "scenes": [...]}`` as JSON, creating folders."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"summary": summary, "scenes": [asdict(r) for r in results]}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def _fmt(value: float | None, spec: str = ".1f") -> str:
    return "-" if value is None else format(value, spec)


def format_report(summary: dict[str, Any]) -> str:
    """Human-readable summary."""

    def line(label: str, s: dict[str, Any]) -> str:
        rate = None if s["hit_rate"] is None else 100 * s["hit_rate"]
        return (
            f"{label:<8} scenes {s['scenes']:>2}  hits {s['hits']:>2}  "
            f"hit rate {_fmt(rate, '.0f')}%  mean distance "
            f"{_fmt(s['mean_distance_px'])} px  median "
            f"{_fmt(s['median_distance_px'])} px"
        )

    lines = [
        f"Precision bench: {summary['model']} (effort {summary['effort']})",
        line("overall", summary),
        *(line(kind, s) for kind, s in summary["by_kind"].items()),
        f"calls {summary['calls']}  cost ${summary['cost_usd']:.4f}  "
        f"no click chosen {summary['no_point']}  "
        f"hits without zoom {summary['coarse_hits']}",
    ]
    if summary["stopped_by_budget"]:
        lines.append("Stopped early: the budget was reached.")
    return "\n".join(lines)


# --------------------------------------------------------------------- main


def _load_env(env_file: Path) -> None:
    """Load ``env_file`` into ``os.environ`` without overriding existing values."""

    from dotenv import dotenv_values

    for key, value in dotenv_values(env_file).items():
        if value is not None:
            os.environ.setdefault(key, value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.precision", description=__doc__)
    parser.add_argument("--scenes", type=int, default=None, help="first N scenes")
    parser.add_argument("--model", default=None, help="default: GUI_AGENT_MODEL")
    parser.add_argument("--effort", default=None, help="default: GUI_AGENT_EFFORT")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET, help="USD cap")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, default=None, help="result JSON path")
    parser.add_argument(
        "--baseline", action="store_true", help=f"also write {BASELINE_PATH.name}"
    )
    args = parser.parse_args(argv)

    if args.env_file.is_file():
        _load_env(args.env_file)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set (environment or --env-file).",
            file=sys.stderr,
        )
        return 2

    from agent.planner import DEFAULT_EFFORT, DEFAULT_MODEL, Planner

    model = args.model or os.environ.get("GUI_AGENT_MODEL") or DEFAULT_MODEL
    effort = args.effort or os.environ.get("GUI_AGENT_EFFORT") or DEFAULT_EFFORT
    scenes = all_scenes()[: args.scenes]
    planner = Planner(model=model, effort=effort)

    results, stopped = run_bench(scenes, planner, args.budget)
    summary = summarize(results, model, effort, stopped)
    print(format_report(summary))

    path = save_results(results, summary, args.output or default_result_path(model))
    print(f"Saved {path}")
    if args.baseline:
        print(f"Saved {save_results(results, summary, BASELINE_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
