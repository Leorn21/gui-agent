"""Emergency pause / stop for the GUI agent.

A global hotkey listener (pynput) lets the user take back control at any time:

- Pause hotkey (default Ctrl+Alt+Q) while running: pause. The controller calls
  ``check()`` between keystrokes and mouse moves, so the action in progress is
  cut short.
- While paused, resume hotkey (default Ctrl+Alt+R): resume. The state machine
  discards the interrupted action and perceives the screen again.
- While paused, pause hotkey again: abort for good. The abort is latched.

This module only tracks the state and reports changes through a callback;
showing them to the user (console message, beep) is the caller's job.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from enum import StrEnum
from types import TracebackType
from typing import Protocol, Self

from pynput import keyboard

DEFAULT_PAUSE_HOTKEY = "ctrl+alt+q"
DEFAULT_RESUME_HOTKEY = "ctrl+alt+r"
DEFAULT_DEBOUNCE = 0.5  # seconds; absorbs keyboard auto-repeat on a held hotkey

logger = logging.getLogger(__name__)


class SafetyState(StrEnum):
    """Run state controlled by the user's hotkeys."""

    RUNNING = "running"
    PAUSED = "paused"
    ABORTED = "aborted"


class PausedError(Exception):
    """The user paused the agent; the action in progress must stop.

    Deliberately not an ``InterruptedError``: after a pause the agent resumes
    by perceiving again, while an abort ends the run.
    """


class HotkeyListener(Protocol):
    """The part of ``pynput.keyboard.GlobalHotKeys`` this module uses."""

    def start(self) -> None: ...

    def stop(self) -> None: ...


ListenerFactory = Callable[[dict[str, Callable[[], None]]], HotkeyListener]


def to_pynput_hotkey(text: str) -> str:
    """Convert ``"ctrl+alt+q"`` style text to pynput's ``"<ctrl>+<alt>+q"``.

    Case and spaces are ignored; single characters stay bare and named keys
    (``ctrl``, ``f12``...) are wrapped in angle brackets.

    Raises:
        ValueError: If the text is empty, has an empty part or names an
            unknown key.
    """

    parts = [part.strip().lower() for part in text.split("+")]
    if not text.strip() or any(not part for part in parts):
        raise ValueError(f"Invalid hotkey: {text!r}")

    combo = "+".join(part if len(part) == 1 else f"<{part}>" for part in parts)
    try:
        keyboard.HotKey.parse(combo)
    except ValueError as exc:
        raise ValueError(f"Invalid hotkey: {text!r}") from exc
    return combo


def _default_listener(hotkeys: dict[str, Callable[[], None]]) -> HotkeyListener:
    return keyboard.GlobalHotKeys(hotkeys)


class EmergencyKillSwitch:
    """Thread-safe pause/resume/abort switch driven by global hotkeys.

    Use it as a context manager, or call ``start()`` and ``stop()``.
    """

    def __init__(
        self,
        pause_hotkey: str = DEFAULT_PAUSE_HOTKEY,
        resume_hotkey: str = DEFAULT_RESUME_HOTKEY,
        on_state_change: Callable[[SafetyState], None] | None = None,
        debounce: float = DEFAULT_DEBOUNCE,
        listener_factory: ListenerFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        Args:
            pause_hotkey: Pauses while running, aborts while paused.
            resume_hotkey: Resumes while paused.
            on_state_change: Called with the new state after every real
                change, from the listener thread for hotkeys. Exceptions it
                raises are logged and ignored.
            debounce: Seconds after a state change during which hotkeys are
                ignored. Programmatic calls are never debounced.
            listener_factory: Builds the hotkey listener (tests inject a fake).
            clock: Monotonic time source for the debounce.

        Raises:
            ValueError: If a hotkey is invalid, both hotkeys are the same or
                ``debounce`` is negative.
        """

        pause = to_pynput_hotkey(pause_hotkey)
        resume = to_pynput_hotkey(resume_hotkey)
        if pause == resume:
            raise ValueError("Pause and resume hotkeys must be different")
        if debounce < 0:
            raise ValueError(f"debounce must not be negative, got {debounce}")

        self._hotkeys = {pause: self._on_pause_hotkey, resume: self._on_resume_hotkey}
        self._on_state_change = on_state_change
        self._debounce = debounce
        self._listener_factory = listener_factory or _default_listener
        self._clock = clock

        self._condition = threading.Condition()
        self._state = SafetyState.RUNNING
        self._last_change = -math.inf
        self._listener: HotkeyListener | None = None
        self._stopped = False

    # ------------------------------------------------------------------ state

    @property
    def state(self) -> SafetyState:
        """Current state."""

        return self._state

    def is_paused(self) -> bool:
        return self._state is SafetyState.PAUSED

    def is_aborted(self) -> bool:
        return self._state is SafetyState.ABORTED

    def check(self) -> None:
        """Cheap guard for the controller and the state machine.

        Raises:
            PausedError: If paused.
            InterruptedError: If aborted.
        """

        state = self._state
        if state is SafetyState.PAUSED:
            raise PausedError("Paused by the user")
        if state is SafetyState.ABORTED:
            raise InterruptedError("Aborted by the user")

    def wait_while_paused(self, timeout: float | None = None) -> bool:
        """Block while paused.

        Returns:
            ``True`` once running, ``False`` if ``timeout`` expired while
            still paused.

        Raises:
            InterruptedError: If the switch is or becomes aborted.
        """

        with self._condition:
            self._condition.wait_for(
                lambda: self._state is not SafetyState.PAUSED, timeout
            )
            if self._state is SafetyState.ABORTED:
                raise InterruptedError("Aborted by the user")
            return self._state is SafetyState.RUNNING

    # ------------------------------------------------------------- transitions

    def pause(self) -> None:
        """Pause if running."""

        self._transition({SafetyState.RUNNING: SafetyState.PAUSED}, debounced=False)

    def resume(self) -> None:
        """Resume if paused."""

        self._transition({SafetyState.PAUSED: SafetyState.RUNNING}, debounced=False)

    def abort(self) -> None:
        """Abort for good, from running or paused."""

        self._transition(
            {
                SafetyState.RUNNING: SafetyState.ABORTED,
                SafetyState.PAUSED: SafetyState.ABORTED,
            },
            debounced=False,
        )

    def _on_pause_hotkey(self) -> None:
        self._transition(
            {
                SafetyState.RUNNING: SafetyState.PAUSED,
                SafetyState.PAUSED: SafetyState.ABORTED,
            },
            debounced=True,
        )

    def _on_resume_hotkey(self) -> None:
        self._transition({SafetyState.PAUSED: SafetyState.RUNNING}, debounced=True)

    def _transition(
        self, moves: dict[SafetyState, SafetyState], *, debounced: bool
    ) -> None:
        with self._condition:
            now = self._clock()
            if debounced and now - self._last_change < self._debounce:
                return
            new_state = moves.get(self._state)
            if new_state is None:
                return
            self._state = new_state
            self._last_change = now
            self._condition.notify_all()

        if self._on_state_change is not None:
            try:
                self._on_state_change(new_state)
            except Exception:
                logger.exception("on_state_change callback failed")

    # --------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Start listening for the hotkeys (once)."""

        if self._listener is None:
            self._listener = self._listener_factory(self._hotkeys)
            self._listener.start()

    def stop(self) -> None:
        """Stop the listener. Safe to call more than once."""

        if self._listener is not None and not self._stopped:
            self._listener.stop()
            self._stopped = True

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()


__all__ = [
    "DEFAULT_DEBOUNCE",
    "DEFAULT_PAUSE_HOTKEY",
    "DEFAULT_RESUME_HOTKEY",
    "EmergencyKillSwitch",
    "PausedError",
    "SafetyState",
    "to_pynput_hotkey",
]
