import pytest
from pydantic import ValidationError

from schemas.actions import ActionType, AgentAction, Coordinate


def test_valid_click_action():
    action = AgentAction(
        thought="Click target",
        action_type=ActionType.CLICK,
        coordinate=Coordinate(x=500, y=300),
    )
    assert action.action_type == ActionType.CLICK
    assert action.coordinate.x == 500


def test_invalid_click_missing_coordinates():
    with pytest.raises(ValidationError):
        AgentAction(thought="No coords", action_type=ActionType.CLICK)


def test_type_action_requires_text():
    with pytest.raises(ValidationError):
        AgentAction(thought="Typing nothing", action_type=ActionType.TYPE)


def test_scroll_action_with_target_and_amount():
    action = AgentAction(
        thought="Scroll the list down",
        action_type=ActionType.SCROLL,
        target_id=42,
        scroll_amount=3,
    )
    assert action.scroll_amount == 3


@pytest.mark.parametrize(
    "fields",
    [
        {"scroll_amount": 3},  # no target
        {"target_id": 42},  # no amount
        {"target_id": 42, "scroll_amount": 0},  # zero amount
    ],
)
def test_scroll_requires_target_and_nonzero_amount(fields):
    with pytest.raises(ValidationError):
        AgentAction(thought="Scroll", action_type=ActionType.SCROLL, **fields)


def test_finish_action_sets_terminal():
    action = AgentAction(thought="Done", action_type=ActionType.FINISH)
    assert action.is_terminal is True


def test_drag_between_grid_cells():
    action = AgentAction(
        thought="Move the file to the folder",
        action_type=ActionType.DRAG,
        target_id=12,
        drag_end_target_id=57,
    )
    assert (action.target_id, action.drag_end_target_id) == (12, 57)


def test_drag_between_coordinates_still_works():
    action = AgentAction(
        thought="Drag",
        action_type=ActionType.DRAG,
        coordinate=Coordinate(x=10, y=10),
        drag_end_coordinate=Coordinate(x=50, y=50),
    )
    assert action.drag_payload is not None


@pytest.mark.parametrize(
    "fields",
    [
        {"target_id": 12},  # no end
        {"drag_end_target_id": 57},  # no start
        {"target_id": 12, "drag_end_coordinate": Coordinate(x=5, y=5)},  # mixed
    ],
)
def test_drag_needs_two_cells_or_two_coordinates(fields):
    with pytest.raises(ValidationError):
        AgentAction(thought="Drag", action_type=ActionType.DRAG, **fields)


def test_drag_end_target_id_must_not_be_negative():
    with pytest.raises(ValidationError):
        AgentAction(
            thought="Drag",
            action_type=ActionType.DRAG,
            target_id=1,
            drag_end_target_id=-1,
        )
