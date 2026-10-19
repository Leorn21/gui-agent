"""The agent loop: perceive, decide, act, evaluate.

Each step captures the primary monitor, draws the coarse grid, asks the
planner for an action, refines its target on an enlarged region (zoom), acts
through the controller and waits for the screen to change. The frame left by
that wait is the next step's perception.

The emergency switch is checked at the start of every step and right before
acting (the controller also checks it between movements). A pause discards
the step in progress and, after resuming, the agent perceives again; an abort
ends the run.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from agent.prompts import StepRecord
from controller.base import BaseController
from controller.safety import PausedError
from schemas.actions import ActionType, AgentAction
from vision.capture import Frame
from vision.diff import FrameDiff
from vision.som_tagger import TaggedFrame, VisualTagger

DEFAULT_MAX_STEPS = 10
DEFAULT_STUCK_LIMIT = 3
DEFAULT_MAX_WAIT = 10.0
DEFAULT_ZOOM_GRID = 8
DEFAULT_ZOOM_MARGIN = 0.5  # cells captured around the chosen one, per side
WAIT_SLICE = 0.1  # seconds between safety checks while waiting

# USD per million tokens: (input, output). Cache writes cost 1.25x input,
# cache reads 0.1x input.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

_CLICKS = (ActionType.CLICK, ActionType.DOUBLE_CLICK, ActionType.RIGHT_CLICK)


class RunStatus(StrEnum):
    """How a run ended."""

    FINISHED = "finished"
    FAILED = "failed"
    ABORTED = "aborted"
    MAX_STEPS = "max_steps"
    STUCK = "stuck"


@dataclass(frozen=True)
class StepEvent:
    """One executed step, as reported to ``on_step`` and the recorder.

    Attributes:
        number: Step number, from 1.
        action: The action taken.
        point: Screen point acted on (start point for a drag), or ``None``.
        end_point: End point of a drag, or ``None``.
        screen_changed: Whether the screen visibly changed afterwards.
        changed_area: ``(x, y, width, height)`` of the change, or ``None``.
        cost_usd: Estimated API cost of the run so far.
    """

    number: int
    action: AgentAction
    point: tuple[int, int] | None
    end_point: tuple[int, int] | None
    screen_changed: bool
    changed_area: tuple[int, int, int, int] | None
    cost_usd: float


@dataclass(frozen=True)
class RunResult:
    """Outcome of a run."""

    status: RunStatus
    reason: str
    steps: tuple[StepRecord, ...]
    cost_usd: float
    run_dir: Path | None = None


class Screen(Protocol):
    """What the loop needs from ``vision.capture.ScreenCapture``."""

    def capture(self) -> Frame: ...

    def capture_region(self, x: int, y: int, width: int, height: int) -> Frame: ...


def estimate_cost(usages: Sequence[Any], model: str) -> float:
    """Estimated USD cost of API calls from their ``usage`` objects.

    Returns 0 for a model missing from ``PRICES``.
    """

    if model not in PRICES:
        return 0.0
    price_in, price_out = (p / 1_000_000 for p in PRICES[model])
    total = 0.0
    for usage in usages:
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        total += usage.input_tokens * price_in
        total += cache_write * price_in * 1.25
        total += cache_read * price_in * 0.1
        total += usage.output_tokens * price_out
    return total


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


class AgentStateMachine:
    """Runs a goal to completion with a planner, a screen and a controller."""

    def __init__(
        self,
        planner: Any,
        controller: BaseController,
        screen: Screen,
        safety: Any,
        differ: FrameDiff | None = None,
        tagger: VisualTagger | None = None,
        recorder: Any = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        stuck_limit: int = DEFAULT_STUCK_LIMIT,
        max_wait: float = DEFAULT_MAX_WAIT,
        zoom_grid: int = DEFAULT_ZOOM_GRID,
        zoom_margin: float = DEFAULT_ZOOM_MARGIN,
        on_step: Callable[[StepEvent], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """
        Args:
            planner: ``agent.planner.Planner`` (``decide``, ``refine``,
                ``usage``, ``model``).
            controller: Executes actions in screen pixels.
            screen: ``vision.capture.ScreenCapture`` of the primary monitor.
            safety: ``controller.safety.EmergencyKillSwitch``.
            differ: Detects screen changes; a ``FrameDiff`` by default.
            tagger: Draws the grids; a ``VisualTagger`` by default.
            recorder: Optional ``agent.recorder.RunRecorder``.
            max_steps: Executed actions before stopping.
            stuck_limit: Consecutive actions without visible change (waits
                excluded) before stopping.
            max_wait: Cap in seconds for a WAIT action.
            zoom_grid: Columns and rows of the zoom grid.
            zoom_margin: Cells of context around the chosen cell on each
                side of the zoom, in ``(0, 2]``. 0.5 captures 2x2 cells.
            on_step: Called after every executed step.
            sleep: Injection point for tests.

        Raises:
            ValueError: On out-of-range settings.
        """

        if max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, got {max_steps}")
        if stuck_limit < 1:
            raise ValueError(f"stuck_limit must be at least 1, got {stuck_limit}")
        if max_wait < 0:
            raise ValueError(f"max_wait must not be negative, got {max_wait}")
        if zoom_grid < 2:
            raise ValueError(f"zoom_grid must be at least 2, got {zoom_grid}")
        if not 0 < zoom_margin <= 2:
            raise ValueError(f"zoom_margin must be in (0, 2], got {zoom_margin}")

        self._planner = planner
        self._controller = controller
        self._screen = screen
        self._safety = safety
        self._differ = differ or FrameDiff()
        self._tagger = tagger or VisualTagger()
        self._recorder = recorder
        self._max_steps = max_steps
        self._stuck_limit = stuck_limit
        self._max_wait = max_wait
        self._zoom_grid = zoom_grid
        self._zoom_margin = zoom_margin
        self._on_step = on_step
        self._sleep = sleep

    def run(self, goal: str) -> RunResult:
        """Work on ``goal`` until it finishes, fails, aborts or hits a limit."""

        run_dir = self._recorder.start(goal) if self._recorder else None
        history: list[StepRecord] = []
        frame: Frame | None = None
        stuck = 0
        status: RunStatus | None = None
        reason = ""

        try:
            while status is None:
                try:
                    self._safety.check()
                    if frame is None:
                        frame = self._screen.capture()
                    tagged = self._tagger.apply_labeled_grid(frame)
                    action = self._planner.decide(goal, tagged, history)
                    if action.action_type is ActionType.FINISH:
                        status, reason = RunStatus.FINISHED, action.thought
                        break
                    if action.action_type is ActionType.FAIL:
                        status, reason = RunStatus.FAILED, action.thought
                        break
                    point, end_point = self._locate(goal, action, tagged)
                    self._safety.check()
                    self._execute(action, point, end_point)
                except PausedError:
                    self._safety.wait_while_paused()  # raises if aborted meanwhile
                    frame = None  # the screen may have changed: perceive again
                    continue

                diff, frame = self._differ.wait_for_change(frame, self._screen.capture)
                area = diff.bounds
                record = StepRecord(
                    action,
                    screen_changed=diff.changed,
                    changed_area=None
                    if area is None
                    else (area.x, area.y, area.width, area.height),
                )
                history.append(record)
                if action.action_type is not ActionType.WAIT:
                    stuck = 0 if diff.changed else stuck + 1

                event = StepEvent(
                    number=len(history),
                    action=action,
                    point=point,
                    end_point=end_point,
                    screen_changed=diff.changed,
                    changed_area=record.changed_area,
                    cost_usd=self._cost(),
                )
                if self._recorder:
                    self._recorder.record_step(
                        event, self._annotate(tagged, point, end_point)
                    )
                if self._on_step:
                    self._on_step(event)

                if stuck >= self._stuck_limit:
                    status = RunStatus.STUCK
                    reason = f"{stuck} actions in a row without visible change"
                elif len(history) >= self._max_steps:
                    status = RunStatus.MAX_STEPS
                    reason = f"Reached the maximum of {self._max_steps} steps"
        except InterruptedError:
            status, reason = RunStatus.ABORTED, "Aborted by the user"

        result = RunResult(
            status=status,
            reason=reason,
            steps=tuple(history),
            cost_usd=self._cost(),
            run_dir=run_dir,
        )
        if self._recorder:
            self._recorder.finish(result)
        return result

    # ----------------------------------------------------------------- steps

    def _locate(
        self, goal: str, action: AgentAction, tagged: TaggedFrame
    ) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        """Screen point(s) the action targets, refined with the zoom."""

        kind = action.action_type
        if kind in _CLICKS or kind is ActionType.SCROLL:
            return self._point(
                goal, action, tagged, action.target_id, action.coordinate
            ), None
        if kind is ActionType.DRAG:
            start = self._point(
                goal, action, tagged, action.target_id, action.coordinate
            )
            end = self._point(
                goal,
                action,
                tagged,
                action.drag_end_target_id,
                action.drag_end_coordinate,
                drag_end=True,
            )
            return start, end
        return None, None

    def _point(
        self,
        goal: str,
        action: AgentAction,
        tagged: TaggedFrame,
        cell: int | None,
        coordinate: Any,
        drag_end: bool = False,
    ) -> tuple[int, int]:
        if cell is None:  # coordinates in image pixels (not produced by the planner)
            return tagged.frame.to_screen(coordinate.x, coordinate.y)

        zoom = self._zoom_region(tagged, cell)
        zoom_tagged = self._tagger.apply_labeled_grid(
            zoom, grid_cols=self._zoom_grid, grid_rows=self._zoom_grid
        )
        fine = self._planner.refine(goal, action, zoom_tagged, drag_end=drag_end)
        if fine is None:
            return tagged.cell_to_screen(cell)
        return zoom_tagged.cell_to_screen(fine)

    def _zoom_region(self, tagged: TaggedFrame, cell: int) -> Frame:
        """Capture the cell plus ``zoom_margin`` cells around it at full resolution."""

        frame = tagged.frame
        width, height = frame.size
        x0, y0, x1, y1 = tagged.cell_bounds(cell)
        cell_w, cell_h = x1 - x0, y1 - y0
        pad_x, pad_y = cell_w * self._zoom_margin, cell_h * self._zoom_margin
        ix0, iy0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
        ix1, iy1 = min(width, x1 + pad_x), min(height, y1 + pad_y)

        # Image pixels -> pixels relative to the monitor (capture_region's space).
        left = _round_half_up(ix0 / frame.scale)
        top = _round_half_up(iy0 / frame.scale)
        right = _round_half_up(ix1 / frame.scale)
        bottom = _round_half_up(iy1 / frame.scale)
        return self._screen.capture_region(left, top, right - left, bottom - top)

    def _execute(
        self,
        action: AgentAction,
        point: tuple[int, int] | None,
        end_point: tuple[int, int] | None,
    ) -> None:
        kind = action.action_type
        if kind is ActionType.CLICK:
            self._controller.click(*point)
        elif kind is ActionType.DOUBLE_CLICK:
            self._controller.double_click(*point)
        elif kind is ActionType.RIGHT_CLICK:
            self._controller.click(*point, button="right")
        elif kind is ActionType.TYPE:
            self._controller.type_text(action.text)
        elif kind is ActionType.PRESS_KEY:
            self._controller.press_key(action.key)
        elif kind is ActionType.DRAG:
            self._controller.drag(*point, *end_point, action.duration)
        elif kind is ActionType.SCROLL:
            self._controller.scroll(*point, action.scroll_amount)
        elif kind is ActionType.WAIT:
            self._wait(min(action.duration, self._max_wait))

    def _wait(self, seconds: float) -> None:
        remaining = seconds
        while remaining > 1e-9:
            self._safety.check()
            chunk = min(WAIT_SLICE, remaining)
            self._sleep(chunk)
            remaining -= chunk

    # --------------------------------------------------------------- helpers

    def _cost(self) -> float:
        return estimate_cost(self._planner.usage, self._planner.model)

    @staticmethod
    def _annotate(
        tagged: TaggedFrame,
        point: tuple[int, int] | None,
        end_point: tuple[int, int] | None,
    ) -> NDArray[np.uint8]:
        """The coarse grid image with the acted-on point(s) marked in red."""

        image = tagged.image.copy()
        frame = tagged.frame
        for p in (point, end_point):
            if p is None:
                continue
            x = _round_half_up((p[0] - frame.origin[0]) * frame.scale)
            y = _round_half_up((p[1] - frame.origin[1]) * frame.scale)
            cv2.circle(image, (x, y), 10, (0, 0, 255), 2)
            cv2.drawMarker(image, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        return image


__all__ = [
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MAX_WAIT",
    "DEFAULT_STUCK_LIMIT",
    "DEFAULT_ZOOM_GRID",
    "DEFAULT_ZOOM_MARGIN",
    "PRICES",
    "AgentStateMachine",
    "RunResult",
    "RunStatus",
    "StepEvent",
    "estimate_cost",
]
