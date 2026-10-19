"""Operating-system input controller built on pynput.

Moves the real mouse and types on the real keyboard. Every step of a long
action (each stretch of a mouse trajectory, each typed character) first calls
the emergency switch's ``check()``, so a pause or abort cuts it short; held
mouse buttons and keys are always released on the way out.
"""

from __future__ import annotations

import ctypes
import sys
import time
from collections.abc import Callable, Iterable
from typing import Protocol

import mss
from pynput import keyboard, mouse

from controller.base import BaseController
from controller.safety import (
    DEFAULT_PAUSE_HOTKEY,
    DEFAULT_RESUME_HOTKEY,
    to_pynput_hotkey,
)

DEFAULT_MOVE_DURATION = 0.2
DEFAULT_MOVE_STEPS = 10
DEFAULT_TYPE_INTERVAL = 0.02

BUTTONS = {
    "left": mouse.Button.left,
    "right": mouse.Button.right,
    "middle": mouse.Button.middle,
}


class SafetyCheck(Protocol):
    """Anything with ``check()``, e.g. ``controller.safety.EmergencyKillSwitch``."""

    def check(self) -> None: ...


def enable_dpi_awareness() -> None:
    """Make this process use physical pixels on Windows (no-op elsewhere).

    Without it, Windows scales mouse coordinates on displays set to 125% or
    150%, and they stop matching the pixels ``mss`` captures.
    """

    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor aware
    except AttributeError, OSError:
        ctypes.windll.user32.SetProcessDPIAware()


def virtual_screen_bounds() -> tuple[int, int, int, int]:
    """``(left, top, width, height)`` of all monitors together, from mss."""

    with mss.MSS() as sct:
        area = sct.monitors[0]
    return area["left"], area["top"], area["width"], area["height"]


class OSController(BaseController):
    """Real mouse and keyboard control through pynput."""

    def __init__(
        self,
        safety: SafetyCheck | None = None,
        move_duration: float = DEFAULT_MOVE_DURATION,
        move_steps: int = DEFAULT_MOVE_STEPS,
        type_interval: float = DEFAULT_TYPE_INTERVAL,
        reserved_hotkeys: Iterable[str] = (DEFAULT_PAUSE_HOTKEY, DEFAULT_RESUME_HOTKEY),
        screen_bounds: tuple[int, int, int, int] | None = None,
        mouse_controller: mouse.Controller | None = None,
        keyboard_controller: keyboard.Controller | None = None,
        sleep: Callable[[float], None] = time.sleep,
        make_dpi_aware: Callable[[], None] = enable_dpi_awareness,
    ) -> None:
        """
        Args:
            safety: Checked before every step; ``None`` disables the checks.
            move_duration: Seconds a mouse movement takes (0 = jump).
            move_steps: Stretches a movement is split into.
            type_interval: Seconds between typed characters.
            reserved_hotkeys: Combos ``press_key`` refuses (the safety
                hotkeys, so the agent cannot pause or resume itself).
            screen_bounds: ``(left, top, width, height)`` of the virtual
                screen; read from mss when omitted.
            mouse_controller, keyboard_controller, sleep, make_dpi_aware:
                Injection points for tests.

        Raises:
            ValueError: On negative durations or ``move_steps < 1``.
        """

        if move_duration < 0:
            raise ValueError(f"move_duration must not be negative, got {move_duration}")
        if move_steps < 1:
            raise ValueError(f"move_steps must be at least 1, got {move_steps}")
        if type_interval < 0:
            raise ValueError(f"type_interval must not be negative, got {type_interval}")

        # Before anything reads coordinates, so they are physical pixels.
        make_dpi_aware()

        self._safety = safety
        self._move_duration = move_duration
        self._move_steps = move_steps
        self._type_interval = type_interval
        self._reserved = {to_pynput_hotkey(h) for h in reserved_hotkeys}
        self._bounds = screen_bounds or virtual_screen_bounds()
        self._mouse = mouse_controller or mouse.Controller()
        self._keyboard = keyboard_controller or keyboard.Controller()
        self._sleep = sleep

    # ----------------------------------------------------------------- mouse

    @property
    def position(self) -> tuple[int, int]:
        """Current cursor position."""

        x, y = self._mouse.position
        return int(x), int(y)

    def move_to(self, x: int, y: int) -> None:
        """Move the cursor to ``(x, y)`` along a short trajectory."""

        self._require_on_screen(x, y)
        self._move(x, y, self._move_duration)

    def click(self, x: int, y: int, button: str = "left") -> None:
        self._click(x, y, button, count=1)

    def double_click(self, x: int, y: int) -> None:
        self._click(x, y, "left", count=2)

    def drag(
        self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float
    ) -> None:
        if duration < 0:
            raise ValueError(f"duration must not be negative, got {duration}")
        self._require_on_screen(start_x, start_y)
        self._require_on_screen(end_x, end_y)

        self._move(start_x, start_y, self._move_duration)
        self._check()
        self._mouse.press(mouse.Button.left)
        try:
            self._move(end_x, end_y, duration)
        finally:
            self._mouse.release(mouse.Button.left)

    def scroll(self, x: int, y: int, amount: int) -> None:
        if amount == 0:
            raise ValueError("amount must not be zero")
        self._require_on_screen(x, y)

        self._move(x, y, self._move_duration)
        self._check()
        self._mouse.scroll(0, -amount)  # pynput: positive dy scrolls up

    # -------------------------------------------------------------- keyboard

    def type_text(self, text: str) -> None:
        if not text:
            raise ValueError("text must not be empty")

        for char in text:
            self._check()
            self._keyboard.type(char)
            self._sleep(self._type_interval)

    def press_key(self, key: str) -> None:
        combo = to_pynput_hotkey(key)
        if combo in self._reserved:
            raise ValueError(f"{key!r} is reserved for the emergency switch")
        keys = keyboard.HotKey.parse(combo)

        self._check()
        held = []
        try:
            for k in keys:
                held.append(k)
                self._keyboard.press(k)
        finally:
            for k in reversed(held):
                self._keyboard.release(k)

    # --------------------------------------------------------------- helpers

    def _click(self, x: int, y: int, button: str, count: int) -> None:
        if button not in BUTTONS:
            raise ValueError(f"button must be one of {sorted(BUTTONS)}, got {button!r}")
        self._require_on_screen(x, y)

        self._move(x, y, self._move_duration)
        self._check()
        self._mouse.click(BUTTONS[button], count)

    def _move(self, x: int, y: int, duration: float) -> None:
        start_x, start_y = self.position
        steps = 1 if duration == 0 else self._move_steps
        for i in range(1, steps + 1):
            self._check()
            t = i / steps
            self._mouse.position = (
                round(start_x + (x - start_x) * t),
                round(start_y + (y - start_y) * t),
            )
            if duration:
                self._sleep(duration / steps)

    def _check(self) -> None:
        if self._safety is not None:
            self._safety.check()

    def _require_on_screen(self, x: int, y: int) -> None:
        left, top, width, height = self._bounds
        if not (left <= x < left + width and top <= y < top + height):
            raise ValueError(
                f"Point ({x}, {y}) is outside the screen "
                f"({left}, {top}, {width}x{height})"
            )


__all__ = [
    "DEFAULT_MOVE_DURATION",
    "DEFAULT_MOVE_STEPS",
    "DEFAULT_TYPE_INTERVAL",
    "OSController",
    "SafetyCheck",
    "enable_dpi_awareness",
    "virtual_screen_bounds",
]
