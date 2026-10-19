"""GUI agent entry point.

    python main.py                      # floating command bar
    python main.py --goal "Abrí ..."    # one goal in the console, no window

Settings come from the environment, loaded from ``.env`` (see
``.env.example``). The console always shows each step, the safety switch's
state and the final result with its cost; the command bar shows only the
status.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agent.prompts import describe_action
from agent.state_machine import RunResult, RunStatus, StepEvent
from controller.safety import (
    DEFAULT_PAUSE_HOTKEY,
    DEFAULT_RESUME_HOTKEY,
    SafetyState,
)
from ui.command_bar import DEFAULT_START_DELAY, format_hotkey

DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_EFFORT = "low"
DEFAULT_MAX_STEPS = 20
DEFAULT_ZOOM_MARGIN = 0.5


@dataclass(frozen=True)
class Settings:
    """Runtime settings. The API key itself is never stored here."""

    model: str
    effort: str
    max_steps: int
    pause_hotkey: str
    resume_hotkey: str
    zoom_margin: float
    has_api_key: bool


def load_settings(env: Mapping[str, str]) -> Settings:
    """Read settings from an environment mapping.

    Raises:
        ValueError: If ``GUI_AGENT_MAX_STEPS`` is not a positive integer or
            ``GUI_AGENT_ZOOM_MARGIN`` is not a number in ``(0, 2]``.
    """

    raw_steps = env.get("GUI_AGENT_MAX_STEPS", "") or str(DEFAULT_MAX_STEPS)
    max_steps = int(raw_steps)
    if max_steps < 1:
        raise ValueError(f"GUI_AGENT_MAX_STEPS must be at least 1, got {raw_steps}")
    raw_margin = env.get("GUI_AGENT_ZOOM_MARGIN", "") or str(DEFAULT_ZOOM_MARGIN)
    zoom_margin = float(raw_margin)
    if not 0 < zoom_margin <= 2:
        raise ValueError(f"GUI_AGENT_ZOOM_MARGIN must be in (0, 2], got {raw_margin}")
    return Settings(
        model=env.get("GUI_AGENT_MODEL") or DEFAULT_MODEL,
        effort=env.get("GUI_AGENT_EFFORT") or DEFAULT_EFFORT,
        max_steps=max_steps,
        pause_hotkey=env.get("GUI_AGENT_SAFETY_HOTKEY") or DEFAULT_PAUSE_HOTKEY,
        resume_hotkey=env.get("GUI_AGENT_RESUME_HOTKEY") or DEFAULT_RESUME_HOTKEY,
        zoom_margin=zoom_margin,
        has_api_key=bool(env.get("ANTHROPIC_API_KEY", "").strip()),
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gui-agent",
        description=(
            "Desktop agent driven by Claude. Without --goal, opens the command bar."
        ),
    )
    parser.add_argument("--goal", help="Run this goal in the console, without the bar")
    return parser.parse_args(argv)


def exit_code(result: RunResult) -> int:
    if result.status is RunStatus.FINISHED:
        return 0
    if result.status is RunStatus.ABORTED:
        return 130
    return 1


# ------------------------------------------------------------- console texts


def format_step(event: StepEvent) -> str:
    where = "" if event.point is None else f" en {event.point}"
    change = "cambió la pantalla" if event.screen_changed else "sin cambio"
    return (
        f"[{event.number}] {describe_action(event.action)}{where} · {change} · "
        f"US$ {event.cost_usd:.4f}\n    {event.action.thought}"
    )


def format_result(result: RunResult) -> str:
    lines = [
        f"Resultado: {result.status} — {result.reason}",
        f"Pasos: {len(result.steps)} · Costo: US$ {result.cost_usd:.4f}",
    ]
    if result.run_dir is not None:
        lines.append(f"Registro: {result.run_dir}")
    return "\n".join(lines)


def safety_notice(state: SafetyState, pause_hotkey: str, resume_hotkey: str) -> str:
    if state is SafetyState.PAUSED:
        return (
            f"Pausado: {format_hotkey(resume_hotkey)} reanuda, "
            f"{format_hotkey(pause_hotkey)} detiene"
        )
    if state is SafetyState.RUNNING:
        return "Reanudado"
    return "Detenido por el usuario"


# --------------------------------------------------------------------- wiring


def _beep(kind: str) -> None:
    if sys.platform != "win32":
        return
    import winsound

    sounds = {
        "paused": winsound.MB_ICONEXCLAMATION,
        "running": winsound.MB_OK,
        "aborted": winsound.MB_ICONHAND,
        "done": winsound.MB_ICONASTERISK,
    }
    winsound.MessageBeep(sounds[kind])


class _Session:
    """Builds the real components for one run and releases them afterwards.

    Created in the worker thread: mss and the keyboard hook are used from the
    thread that creates them.
    """

    def __init__(
        self,
        settings: Settings,
        on_step: Callable[[StepEvent], None],
        pill: tuple[int, int, int, int] | None = None,
    ) -> None:
        self._settings = settings
        self._on_step = on_step
        self._pill = pill

    def run(self, goal: str) -> RunResult:
        from agent.planner import Planner
        from agent.recorder import RunRecorder
        from agent.state_machine import AgentStateMachine
        from controller.os_controller import OSController
        from controller.safety import EmergencyKillSwitch
        from ui.command_bar import GuardedController, MaskedScreen
        from vision.capture import ScreenCapture

        s = self._settings

        def on_safety(state: SafetyState) -> None:
            print(safety_notice(state, s.pause_hotkey, s.resume_hotkey), flush=True)
            _beep(str(state))

        safety = EmergencyKillSwitch(
            pause_hotkey=s.pause_hotkey,
            resume_hotkey=s.resume_hotkey,
            on_state_change=on_safety,
        )
        controller: Any = OSController(
            safety=safety, reserved_hotkeys=(s.pause_hotkey, s.resume_hotkey)
        )
        with ScreenCapture() as capture, safety:
            screen: Any = capture
            if self._pill is not None:
                screen = MaskedScreen(capture, self._pill)
                controller = GuardedController(
                    controller, self._pill, on_blocked=lambda msg: print(f"    ({msg})")
                )
            machine = AgentStateMachine(
                planner=Planner(model=s.model, effort=s.effort),
                controller=controller,
                screen=screen,
                safety=safety,
                recorder=RunRecorder(),
                max_steps=s.max_steps,
                zoom_margin=s.zoom_margin,
                on_step=self._on_step,
            )
            return machine.run(goal)


def _print_step(event: StepEvent) -> None:
    print(format_step(event), flush=True)


def _print_done(result: RunResult | None, error: BaseException | None) -> None:
    _beep("done")
    if error is not None:
        print(f"Error: {error!r}", file=sys.stderr, flush=True)
    else:
        print(format_result(result), flush=True)


def run_console(settings: Settings, goal: str) -> int:
    import time

    from controller.os_controller import enable_dpi_awareness

    enable_dpi_awareness()
    print(f"Tarea: {goal}")
    print(
        f"{format_hotkey(settings.pause_hotkey)} pausa · "
        f"{format_hotkey(settings.resume_hotkey)} reanuda · "
        f"{format_hotkey(settings.pause_hotkey)} dos veces detiene",
        flush=True,
    )
    time.sleep(DEFAULT_START_DELAY)
    result = _Session(settings, _print_step).run(goal)
    _print_done(result, None)
    return exit_code(result)


def run_bar(settings: Settings) -> int:
    from controller.os_controller import enable_dpi_awareness
    from ui.command_bar import CommandBar

    enable_dpi_awareness()  # before the window, so its geometry is in physical px
    print("Barra abierta: escribí la tarea y Enter. Esc la cierra.", flush=True)
    bar = CommandBar(
        build=lambda on_step, pill: _Session(
            settings, lambda e: (_print_step(e), on_step(e)), pill
        ),
        pause_hotkey=settings.pause_hotkey,
        on_done=_print_done,
    )
    bar.run()
    return 0


# ----------------------------------------------------------------------- main


def _utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except AttributeError, ValueError:
            pass


def main(
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    run_console: Callable[[Settings, str], int] = run_console,
    run_bar: Callable[[Settings], int] = run_bar,
) -> int:
    _utf8_console()
    if env is None:
        from dotenv import load_dotenv

        load_dotenv()
        env = os.environ

    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        settings = load_settings(env)
    except ValueError as exc:
        print(f"Configuración inválida: {exc}", file=sys.stderr)
        return 2
    if not settings.has_api_key:
        print(
            "Falta ANTHROPIC_API_KEY. "
            "Copiá .env.example como .env y pegá tu clave ahí.",
            file=sys.stderr,
        )
        return 2

    if args.goal:
        return run_console(settings, args.goal)
    return run_bar(settings)


if __name__ == "__main__":
    sys.exit(main())
