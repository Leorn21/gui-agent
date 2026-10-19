"""Tests for agent/prompts.py: prompt texts and the text history."""

import pytest

from agent.prompts import (
    SYSTEM_PROMPT,
    ZOOM_SYSTEM_PROMPT,
    StepRecord,
    build_step_prompt,
    build_zoom_prompt,
    format_history,
)
from schemas.actions import ActionType, AgentAction


def click(cell=24, thought="Open the save dialog"):
    return AgentAction(thought=thought, action_type=ActionType.CLICK, target_id=cell)


# ---------------------------------------------------------------- system prompts


@pytest.mark.parametrize("action_type", list(ActionType))
def test_system_prompt_explains_every_action(action_type):
    assert action_type.value in SYSTEM_PROMPT


def test_system_prompt_asks_for_grid_cells_not_pixels():
    text = SYSTEM_PROMPT.lower()
    assert "target_id" in text
    assert "drag_end_target_id" in text
    assert "never" in text and "coordinate" in text


def test_system_prompt_forbids_the_safety_hotkeys():
    assert "ctrl+alt+q" in SYSTEM_PROMPT.lower()
    assert "ctrl+alt+r" in SYSTEM_PROMPT.lower()


def test_system_prompt_is_english_and_static():
    # Static text keeps the cacheable prefix byte-identical between calls.
    assert "{" not in SYSTEM_PROMPT
    assert "screenshot" in SYSTEM_PROMPT.lower()


def test_zoom_prompt_asks_only_for_a_cell():
    text = ZOOM_SYSTEM_PROMPT.lower()
    assert "target_id" in text
    assert "zoom" in text or "enlarged" in text
    assert "{" not in ZOOM_SYSTEM_PROMPT


# ------------------------------------------------------------------- history


def test_empty_history():
    assert format_history([]) == "No previous steps."


def test_history_lines_describe_action_and_screen_change():
    history = [
        StepRecord(click(24), screen_changed=True, changed_area=(1150, 300, 600, 320)),
        StepRecord(
            AgentAction(thought="Name it", action_type=ActionType.TYPE, text="notas"),
            screen_changed=False,
        ),
        StepRecord(
            AgentAction(thought="Save", action_type=ActionType.PRESS_KEY, key="enter")
        ),
    ]
    lines = format_history(history).splitlines()

    assert lines[0].startswith("Step 1: CLICK cell 24")
    assert "Open the save dialog" in lines[0]
    assert "screen changed at x=1150, y=300, 600x320" in lines[0]
    assert lines[1].startswith('Step 2: TYPE "notas"')
    assert "no visible change" in lines[1]
    assert lines[2].startswith("Step 3: PRESS_KEY enter")
    assert "change" not in lines[2]  # unknown: not reported


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (
            AgentAction(
                thought="t", action_type=ActionType.SCROLL, target_id=5, scroll_amount=3
            ),
            "SCROLL cell 5 by 3",
        ),
        (
            AgentAction(
                thought="t",
                action_type=ActionType.DRAG,
                target_id=4,
                drag_end_target_id=9,
            ),
            "DRAG cell 4 to cell 9",
        ),
        (
            AgentAction(thought="t", action_type=ActionType.WAIT, duration=1.5),
            "WAIT 1.5 s",
        ),
        (
            AgentAction(thought="t", action_type=ActionType.DOUBLE_CLICK, target_id=7),
            "DOUBLE_CLICK cell 7",
        ),
    ],
)
def test_history_describes_each_action_kind(action, expected):
    assert format_history([StepRecord(action)]).startswith(f"Step 1: {expected}")


def test_history_keeps_only_the_last_steps_with_absolute_numbers():
    history = [StepRecord(click(i, thought=f"step {i}")) for i in range(15)]
    lines = format_history(history, limit=10).splitlines()

    assert lines[0] == "(5 earlier steps omitted)"
    assert lines[1].startswith("Step 6: CLICK cell 5")
    assert lines[-1].startswith("Step 15: CLICK cell 14")
    assert len(lines) == 11


def test_default_history_limit_is_10():
    history = [StepRecord(click(i)) for i in range(12)]

    assert format_history(history).splitlines()[0] == "(2 earlier steps omitted)"


def test_long_thoughts_are_shortened():
    line = format_history([StepRecord(click(thought="x" * 500))])

    assert len(line) < 250
    assert line.endswith("...")


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_history_limit_raises(limit):
    with pytest.raises(ValueError):
        format_history([], limit=limit)


# ------------------------------------------------------------------ user prompts


def test_step_prompt_has_goal_grid_and_history():
    text = build_step_prompt("Guardá el archivo", "Step 1: CLICK cell 3", 20, 12)

    assert "Guardá el archivo" in text
    assert "20 columns x 12 rows" in text
    assert "0 to 239" in text
    assert "Step 1: CLICK cell 3" in text


def test_zoom_prompt_has_goal_action_and_fine_grid():
    text = build_zoom_prompt("Guardá el archivo", click(24), 8, 8)

    assert "Guardá el archivo" in text
    assert "CLICK" in text
    assert "Open the save dialog" in text
    assert "8 columns x 8 rows" in text
    assert "0 to 63" in text


def test_zoom_prompt_names_the_drag_end_when_refining_it():
    drag = AgentAction(
        thought="Move it",
        action_type=ActionType.DRAG,
        target_id=4,
        drag_end_target_id=9,
    )

    assert "start" in build_zoom_prompt("g", drag, 8, 8, drag_end=False).lower()
    assert "end" in build_zoom_prompt("g", drag, 8, 8, drag_end=True).lower()
