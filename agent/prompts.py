"""Prompt texts for the planner and the text form of the step history.

The two system prompts are fixed strings so the cached prompt prefix stays
byte-identical between calls. Everything that changes per step (goal, grid
size, history) goes in the user message built here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from schemas.actions import ActionType, AgentAction

DEFAULT_HISTORY_LIMIT = 10
MAX_THOUGHT_CHARS = 120
MAX_TEXT_CHARS = 60

SYSTEM_PROMPT = """\
You operate a Windows desktop for a user, one action at a time.

Each turn you get a screenshot of the screen (or of part of it) with a numbered \
grid drawn on top: magenta lines split it into cells, and each cell's number is in \
its top-left corner. Cells are numbered row by row from 0. You also get the user's \
goal and a text history of your previous steps, including whether each one visibly \
changed the screen.

Point at things only with grid cells: put the number of the cell that contains the \
center of your target in target_id. Read that number where it is printed, in the \
top-left corner of the cell; do not compute it from pixel positions or row and \
column arithmetic, which is easy to get wrong. Never answer with pixel \
coordinates; leave coordinate and drag_end_coordinate empty. After you choose a \
cell, you will be shown an enlarged view of that area to refine the exact point, \
so pick the cell that best contains the target.

Available actions (action_type):
- CLICK, DOUBLE_CLICK, RIGHT_CLICK: click the cell in target_id.
- TYPE: type the text in text into the focused field. Click the field first if it \
is not focused.
- PRESS_KEY: press a key or combo in key. Only these are allowed: win, enter, \
tab, shift+tab, esc, up, down, left, right, home, end, page_up, page_down, \
backspace, delete, space, and the shortcuts that mean the same in every \
language: ctrl+c, ctrl+v, ctrl+x, ctrl+z, ctrl+a, alt+tab, alt+f4. For any other \
command inside a program, click its menu or button instead. Never press \
ctrl+alt+q or ctrl+alt+r: they belong to the user's emergency switch.
- DRAG: drag from the cell in target_id to the cell in drag_end_target_id.
- SCROLL: scroll at the cell in target_id by scroll_amount wheel notches; positive \
scrolls down, negative scrolls up.
- WAIT: wait duration seconds for something to load.
- FINISH: the goal is complete.
- FAIL: the goal cannot be completed; explain why in thought.

Rules:
- Return exactly one action per turn.
- In thought, briefly say what you see and why this action moves toward the goal.
- Trust the screenshot over your expectations. If the history shows an action had \
no visible change, do not repeat it unchanged; try something else.
- Choose FINISH only when the screenshot shows the goal is done.

Windows tips:
- To open a program, press the Windows key (PRESS_KEY win), type its name and \
press enter. This is faster than hunting for icons.
- Other keyboard shortcuts depend on the program's language: in a Spanish \
program ctrl+n means bold (negrita), not new. That is why they are not allowed.
- If a shortcut had no visible effect, or not the one you expected, do not \
repeat it: use the program's menu or buttons with clicks instead.
- Programs may reopen documents from an earlier session. Never type into an \
existing document; open a new tab or a new document first, for example with \
the "+" button or the File menu.
"""

ZOOM_SYSTEM_PROMPT = """\
You are refining where to point on the screen.

You get an enlarged view of a small area of the screen with a finer numbered grid \
drawn on top, the user's goal, and the action already chosen for this step. Cells \
are numbered row by row from 0, with each number printed in the cell's top-left \
corner. Read the printed number; do not compute it.

Return in target_id the cell that contains the center of the element the action \
is aimed at, and briefly explain your choice in thought. Do not change the action \
itself.
"""


@dataclass(frozen=True)
class StepRecord:
    """One past step as the planner sees it.

    Attributes:
        action: The action taken.
        screen_changed: Whether the screen visibly changed; ``None`` if unknown.
        changed_area: ``(x, y, width, height)`` in screen pixels of the change.
    """

    action: AgentAction
    screen_changed: bool | None = None
    changed_area: tuple[int, int, int, int] | None = None


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def describe_action(action: AgentAction) -> str:
    """Short text for an action, e.g. ``CLICK cell 24`` or ``TYPE "hola"``."""

    kind = action.action_type

    def where(cell: int | None, point) -> str:
        if cell is not None:
            return f"cell {cell}"
        if point is not None:
            return f"({point.x}, {point.y})"
        return "?"

    if kind in (ActionType.CLICK, ActionType.DOUBLE_CLICK, ActionType.RIGHT_CLICK):
        return f"{kind} {where(action.target_id, action.coordinate)}"
    if kind is ActionType.TYPE:
        return f'TYPE "{_shorten(action.text or "", MAX_TEXT_CHARS)}"'
    if kind is ActionType.PRESS_KEY:
        return f"PRESS_KEY {action.key}"
    if kind is ActionType.DRAG:
        start = where(action.target_id, action.coordinate)
        end = where(action.drag_end_target_id, action.drag_end_coordinate)
        return f"DRAG {start} to {end}"
    if kind is ActionType.SCROLL:
        target = where(action.target_id, action.coordinate)
        return f"SCROLL {target} by {action.scroll_amount}"
    if kind is ActionType.WAIT:
        return f"WAIT {action.duration:g} s"
    return str(kind)


def _describe_change(record: StepRecord) -> str:
    if record.screen_changed is None:
        return ""
    if not record.screen_changed:
        return " -> no visible change"
    if record.changed_area is None:
        return " -> screen changed"
    x, y, w, h = record.changed_area
    return f" -> screen changed at x={x}, y={y}, {w}x{h}"


def format_history(
    history: Sequence[StepRecord], limit: int = DEFAULT_HISTORY_LIMIT
) -> str:
    """One line per step, keeping only the last ``limit`` steps.

    Raises:
        ValueError: If ``limit`` is less than 1.
    """

    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    if not history:
        return "No previous steps."

    skipped = max(0, len(history) - limit)
    lines = [f"({skipped} earlier steps omitted)"] if skipped else []
    for number, record in enumerate(history[skipped:], start=skipped + 1):
        thought = _shorten(record.action.thought, MAX_THOUGHT_CHARS)
        lines.append(
            f"Step {number}: {describe_action(record.action)} — {thought}"
            f"{_describe_change(record)}"
        )
    return "\n".join(lines)


def _grid_line(grid_cols: int, grid_rows: int) -> str:
    return (
        f"Grid: {grid_cols} columns x {grid_rows} rows, "
        f"cells 0 to {grid_cols * grid_rows - 1}."
    )


def build_step_prompt(
    goal: str, history_text: str, grid_cols: int, grid_rows: int
) -> str:
    """User message text for choosing the next action."""

    return (
        f"Goal: {goal}\n\n"
        f"{_grid_line(grid_cols, grid_rows)}\n\n"
        f"Previous steps:\n{history_text}\n\n"
        "Choose the next action."
    )


def build_zoom_prompt(
    goal: str,
    action: AgentAction,
    grid_cols: int,
    grid_rows: int,
    drag_end: bool = False,
) -> str:
    """User message text for refining the point of an already chosen action."""

    if action.action_type is ActionType.DRAG:
        point = (
            "the end point of the drag" if drag_end else "the start point of the drag"
        )
    else:
        point = "the element the action is aimed at"
    return (
        f"Goal: {goal}\n\n"
        f"Chosen action: {action.action_type} — {_shorten(action.thought, 200)}\n\n"
        f"{_grid_line(grid_cols, grid_rows)}\n\n"
        f"Return the cell that contains {point}."
    )


__all__ = [
    "DEFAULT_HISTORY_LIMIT",
    "SYSTEM_PROMPT",
    "ZOOM_SYSTEM_PROMPT",
    "StepRecord",
    "build_step_prompt",
    "build_zoom_prompt",
    "describe_action",
    "format_history",
]
