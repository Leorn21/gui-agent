"""Typed action schemas for the GUI agent.

This module defines the structured output contract used by the planner and
execution loop. The models are intentionally strict so invalid agent output is
rejected early and deterministically.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ActionType(StrEnum):
    """Supported agent action kinds."""

    CLICK = "CLICK"
    DOUBLE_CLICK = "DOUBLE_CLICK"
    RIGHT_CLICK = "RIGHT_CLICK"
    TYPE = "TYPE"
    PRESS_KEY = "PRESS_KEY"
    DRAG = "DRAG"
    SCROLL = "SCROLL"
    WAIT = "WAIT"
    FINISH = "FINISH"
    FAIL = "FAIL"


class Coordinate(BaseModel):
    """A pixel coordinate in screen space."""

    model_config = ConfigDict(extra="forbid")

    x: int = Field(ge=0, description="Horizontal pixel coordinate")
    y: int = Field(ge=0, description="Vertical pixel coordinate")


class DragActionPayload(BaseModel):
    """Explicit drag start and end coordinates.

    This helper is useful when callers want to package drag geometry separately
    from the main action model.
    """

    model_config = ConfigDict(extra="forbid")

    start_coordinate: Coordinate = Field(
        description="Drag start coordinate",
    )
    end_coordinate: Coordinate = Field(
        description="Drag end coordinate",
    )


class AgentAction(BaseModel):
    """Structured action produced by the planner.

    The model is intentionally strict and validates that each action type carries
    the minimum data needed by the controller layer.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    thought: str = Field(..., description="Step-by-step reasoning behind the action")
    action_type: ActionType
    target_id: int | None = Field(
        default=None,
        ge=0,
        description="Set-of-Mark grid or element ID",
    )
    coordinate: Coordinate | None = Field(
        default=None,
        description="Direct pixel target",
    )
    drag_end_coordinate: Coordinate | None = Field(
        default=None,
        description="End coordinate for drag action",
    )
    text: str | None = Field(default=None, description="Text payload to type")
    key: str | None = Field(
        default=None,
        description="Special key stroke or combo (e.g. enter, ctrl+s)",
    )
    scroll_amount: int | None = Field(
        default=None,
        description="Wheel notches to scroll; positive scrolls down",
    )
    duration: float = Field(
        default=0.5,
        ge=0.0,
        description="Wait or drag duration in seconds",
    )
    is_terminal: bool = Field(
        default=False,
        description="Flag indicating execution loop termination",
    )

    @model_validator(mode="after")
    def validate_action(self) -> Self:
        """Enforce action-specific requirements and terminal-state behavior."""

        if self.action_type in {
            ActionType.CLICK,
            ActionType.DOUBLE_CLICK,
            ActionType.RIGHT_CLICK,
        }:
            if self.coordinate is None and self.target_id is None:
                raise ValueError(
                    f"{self.action_type} requires either coordinate or target_id"
                )

        if self.action_type is ActionType.SCROLL:
            if self.coordinate is None and self.target_id is None:
                raise ValueError("SCROLL requires either coordinate or target_id")
            if not self.scroll_amount:
                raise ValueError("SCROLL requires a non-zero scroll_amount")

        if self.action_type is ActionType.TYPE:
            if self.text is None or not self.text.strip():
                raise ValueError("TYPE requires non-empty text")

        if self.action_type is ActionType.PRESS_KEY:
            if self.key is None or not self.key.strip():
                raise ValueError("PRESS_KEY requires non-empty key")

        if self.action_type is ActionType.DRAG:
            if self.coordinate is None or self.drag_end_coordinate is None:
                raise ValueError(
                    "DRAG requires both coordinate and drag_end_coordinate"
                )

        if self.action_type in (ActionType.FINISH, ActionType.FAIL):
            object.__setattr__(self, "is_terminal", True)

        return self

    @property
    def drag_payload(self) -> DragActionPayload | None:
        """Return the drag geometry as a dedicated payload when available."""

        if self.coordinate is None or self.drag_end_coordinate is None:
            return None
        return DragActionPayload(
            start_coordinate=self.coordinate,
            end_coordinate=self.drag_end_coordinate,
        )


__all__ = [
    "ActionType",
    "AgentAction",
    "Coordinate",
    "DragActionPayload",
]
