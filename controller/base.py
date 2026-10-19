"""Abstract input controller for the GUI agent.

Controllers receive absolute screen coordinates in physical pixels (the same
space ``vision.capture`` uses). Mapping an ``AgentAction`` or a grid cell to
those coordinates is the state machine's job, not the controller's.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseController(ABC):
    """Mouse and keyboard actions the agent can perform."""

    @abstractmethod
    def click(self, x: int, y: int, button: str = "left") -> None:
        """Move to ``(x, y)`` and click ``button`` (left, right or middle)."""

    @abstractmethod
    def double_click(self, x: int, y: int) -> None:
        """Move to ``(x, y)`` and double-click the left button."""

    @abstractmethod
    def type_text(self, text: str) -> None:
        """Type ``text`` character by character, Unicode included."""

    @abstractmethod
    def press_key(self, key: str) -> None:
        """Press a key or combo such as ``"enter"`` or ``"ctrl+s"``."""

    @abstractmethod
    def drag(
        self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float
    ) -> None:
        """Drag with the left button from start to end over ``duration`` seconds."""

    @abstractmethod
    def scroll(self, x: int, y: int, amount: int) -> None:
        """Scroll ``amount`` wheel notches at ``(x, y)``; positive scrolls down."""


__all__ = ["BaseController"]
