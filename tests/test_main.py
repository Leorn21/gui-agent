"""Tests for main.py: settings, arguments, console texts and mode selection.

The real wiring (screen, keyboard hook, API client) is exercised by hand; here
the two run modes are injected so no window opens and nothing moves.
"""

import pytest

import main
from agent.state_machine import RunResult, RunStatus, StepEvent
from controller.safety import SafetyState
from schemas.actions import ActionType, AgentAction

KEY_ENV = {"ANTHROPIC_API_KEY": "test-key-not-real"}


# ------------------------------------------------------------------ settings


def test_settings_defaults():
    s = main.load_settings(KEY_ENV)

    assert s.model == "claude-haiku-5-5"
    assert s.effort == "low"
    assert s.max_steps == 20
    assert s.pause_hotkey == "ctrl+alt+q"
    assert s.resume_hotkey == "ctrl+alt+r"
    assert s.has_api_key is True


def test_settings_from_env():
    s = main.load_settings(
        {
            **KEY_ENV,
            "GUI_AGENT_MODEL": "claude-sonnet-5-5",
            "GUI_AGENT_EFFORT": "medium",
            "GUI_AGENT_MAX_STEPS": "25",
            "GUI_AGENT_SAFETY_HOTKEY": "ctrl+shift+p",
            "GUI_AGENT_RESUME_HOTKEY": "ctrl+shift+o",
        }
    )

    assert (s.model, s.effort, s.max_steps) == ("claude-sonnet-5-5", "medium", 25)
    assert (s.pause_hotkey, s.resume_hotkey) == ("ctrl+shift+p", "ctrl+shift+o")


def test_missing_or_empty_key():
    assert main.load_settings({}).has_api_key is False
    assert main.load_settings({"ANTHROPIC_API_KEY": "  "}).has_api_key is False


@pytest.mark.parametrize("value", ["abc", "0", "-3"])
def test_invalid_max_steps_raises(value):
    with pytest.raises(ValueError):
        main.load_settings({**KEY_ENV, "GUI_AGENT_MAX_STEPS": value})


def test_settings_never_expose_the_key():
    assert "test-key-not-real" not in repr(main.load_settings(KEY_ENV))


# ----------------------------------------------------------------- arguments


def test_no_arguments_means_the_command_bar():
    assert main.parse_args([]).goal is None


def test_goal_argument():
    assert main.parse_args(["--goal", "Abrí el Bloc de notas"]).goal == (
        "Abrí el Bloc de notas"
    )


# --------------------------------------------------------------------- modes


def run_main(argv, env):
    calls = []
    code = main.main(
        argv,
        env=env,
        run_console=lambda settings, goal: calls.append(("console", goal)) or 0,
        run_bar=lambda settings: calls.append(("bar", None)) or 0,
    )
    return code, calls


def test_without_api_key_explains_and_exits_2(capsys):
    code, calls = run_main([], {})

    assert code == 2
    assert calls == []
    out = capsys.readouterr()
    assert "ANTHROPIC_API_KEY" in out.out + out.err
    assert ".env" in out.out + out.err


def test_goal_runs_in_the_console():
    code, calls = run_main(["--goal", "x"], KEY_ENV)

    assert calls == [("console", "x")]
    assert code == 0


def test_no_goal_opens_the_bar():
    _, calls = run_main([], KEY_ENV)

    assert calls == [("bar", None)]


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (RunStatus.FINISHED, 0),
        (RunStatus.FAILED, 1),
        (RunStatus.STUCK, 1),
        (RunStatus.MAX_STEPS, 1),
        (RunStatus.ABORTED, 130),
    ],
)
def test_exit_code_for_each_status(status, code):
    res = RunResult(status=status, reason="r", steps=(), cost_usd=0.0)
    assert main.exit_code(res) == code


# ------------------------------------------------------------- console texts


def test_step_line_has_number_action_point_change_cost_and_thought():
    action = AgentAction(
        thought="Abro el menú Inicio", action_type=ActionType.CLICK, target_id=230
    )
    event = StepEvent(
        number=2,
        action=action,
        point=(24, 1060),
        end_point=None,
        screen_changed=False,
        changed_area=None,
        cost_usd=0.0012,
    )
    text = main.format_step(event)

    for part in ("2", "CLICK", "(24, 1060)", "sin cambio", "0.0012", "Abro el menú"):
        assert part in text


def test_result_line_has_status_reason_steps_cost_and_folder(tmp_path):
    res = RunResult(
        status=RunStatus.FINISHED,
        reason="El documento quedó guardado",
        steps=(),
        cost_usd=0.0016,
        run_dir=tmp_path,
    )
    text = main.format_result(res)

    for part in ("finished", "El documento quedó guardado", "0.0016", str(tmp_path)):
        assert part in text


@pytest.mark.parametrize(
    ("state", "words"),
    [
        (SafetyState.PAUSED, ["Pausado", "Ctrl+Alt+R", "Ctrl+Alt+Q"]),
        (SafetyState.RUNNING, ["Reanudado"]),
        (SafetyState.ABORTED, ["Detenido"]),
    ],
)
def test_safety_notices(state, words):
    text = main.safety_notice(
        state, pause_hotkey="ctrl+alt+q", resume_hotkey="ctrl+alt+r"
    )

    for word in words:
        assert word in text
