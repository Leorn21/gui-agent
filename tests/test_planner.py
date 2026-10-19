"""Tests for agent/planner.py.

Unit tests inject a fake Anthropic client whose ``messages.parse`` returns
scripted responses (or raises), so no request leaves the machine. The
integration test calls the real API with Claude Haiku 5.5 on a synthetic
screen; it needs ANTHROPIC_API_KEY and costs well under a cent.
"""

import base64
import os

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from agent.planner import (
    DEFAULT_EFFORT,
    DEFAULT_HISTORY_LIMIT,
    DEFAULT_MODEL,
    Planner,
    ZoomChoice,
)
from agent.prompts import SYSTEM_PROMPT, ZOOM_SYSTEM_PROMPT, StepRecord
from schemas.actions import ActionType, AgentAction, Coordinate
from tests import scenes
from vision.som_tagger import VisualTagger


class FakeUsage:
    def __init__(self, input_tokens=2400, output_tokens=150):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeResponse:
    def __init__(self, parsed, stop_reason="end_turn"):
        self.parsed_output = parsed
        self.stop_reason = stop_reason
        self.stop_details = None
        self.usage = FakeUsage()


class FakeMessages:
    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, *script):
        self.messages = FakeMessages(script)


def validation_error() -> ValidationError:
    try:
        AgentAction(thought="bad", action_type=ActionType.CLICK)
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


def ok(parsed):
    return FakeResponse(parsed)


def click(cell=24):
    return AgentAction(thought="Save", action_type=ActionType.CLICK, target_id=cell)


@pytest.fixture
def tagged():
    return VisualTagger().apply_labeled_grid(scenes.notes_app_frame())  # 240 cells


@pytest.fixture
def zoom_tagged():
    region = scenes.region_frame(288, 0, 288, 270)
    return VisualTagger().apply_labeled_grid(region, grid_cols=8, grid_rows=8)


def user_text(call) -> str:
    return next(
        b["text"] for b in call["messages"][0]["content"] if b["type"] == "text"
    )


def user_image(call) -> np.ndarray:
    block = next(b for b in call["messages"][0]["content"] if b["type"] == "image")
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "image/png"
    raw = base64.b64decode(block["source"]["data"])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


# ------------------------------------------------------------------ settings


def test_defaults():
    assert DEFAULT_MODEL == "claude-haiku-5-5"
    assert DEFAULT_EFFORT == "low"
    assert DEFAULT_HISTORY_LIMIT == 10


@pytest.mark.parametrize(
    "kwargs", [{"effort": "turbo"}, {"max_retries": -1}, {"history_limit": 0}]
)
def test_invalid_settings_raise(kwargs):
    with pytest.raises(ValueError):
        Planner(client=FakeClient(), **kwargs)


# -------------------------------------------------------------------- decide


def test_decide_sends_one_structured_request(tagged):
    client = FakeClient(ok(click()))
    Planner(client=client).decide("Guardá el archivo", tagged)
    (call,) = client.messages.calls

    assert call["model"] == "claude-haiku-5-5"
    assert call["output_format"] is AgentAction
    assert call["output_config"] == {"effort": "low"}
    assert call["max_tokens"] > 0
    assert len(call["messages"]) == 1 and call["messages"][0]["role"] == "user"


def test_system_prompt_is_cached(tagged):
    client = FakeClient(ok(click()))
    Planner(client=client).decide("g", tagged)
    (system,) = client.messages.calls[0]["system"]

    assert system["text"] == SYSTEM_PROMPT
    assert system["cache_control"] == {"type": "ephemeral"}


def test_sends_the_tagged_screenshot_losslessly(tagged):
    client = FakeClient(ok(click()))
    Planner(client=client).decide("g", tagged)

    assert np.array_equal(user_image(client.messages.calls[0]), tagged.image)


def test_prompt_has_goal_grid_and_history(tagged):
    client = FakeClient(ok(click()))
    history = [StepRecord(click(3), screen_changed=False)]
    Planner(client=client).decide("Guardá el archivo", tagged, history)
    text = user_text(client.messages.calls[0])

    assert "Guardá el archivo" in text
    assert "20 columns x 12 rows" in text
    assert "Step 1: CLICK cell 3" in text


def test_history_limit_is_applied(tagged):
    client = FakeClient(ok(click()))
    history = [StepRecord(click(i)) for i in range(5)]
    Planner(client=client, history_limit=3).decide("g", tagged, history)

    assert "(2 earlier steps omitted)" in user_text(client.messages.calls[0])


def test_returns_the_parsed_action(tagged):
    action = click(24)
    assert Planner(client=FakeClient(ok(action))).decide("g", tagged) is action


def test_model_and_effort_are_configurable(tagged):
    client = FakeClient(ok(click()))
    Planner(client=client, model="claude-sonnet-5-5", effort="medium").decide(
        "g", tagged
    )

    assert client.messages.calls[0]["model"] == "claude-sonnet-5-5"
    assert client.messages.calls[0]["output_config"] == {"effort": "medium"}


# ------------------------------------------------------------ invalid answers


def test_retries_once_after_a_validation_error(tagged):
    client = FakeClient(validation_error(), ok(click(30)))
    action = Planner(client=client).decide("g", tagged)

    assert action.target_id == 30
    assert len(client.messages.calls) == 2
    assert "rejected" in user_text(client.messages.calls[1]).lower()


def test_fails_after_the_retry_is_also_invalid(tagged):
    client = FakeClient(validation_error(), validation_error())
    action = Planner(client=client).decide("g", tagged)

    assert action.action_type is ActionType.FAIL
    assert action.is_terminal is True
    assert "invalid" in action.thought.lower()


def test_no_retry_when_max_retries_is_zero(tagged):
    client = FakeClient(validation_error())
    action = Planner(client=client, max_retries=0).decide("g", tagged)

    assert action.action_type is ActionType.FAIL
    assert len(client.messages.calls) == 1


@pytest.mark.parametrize(
    "bad",
    [
        AgentAction(
            thought="pixels",
            action_type=ActionType.CLICK,
            coordinate=Coordinate(x=5, y=5),
        ),
        click(240),  # outside the 20x12 grid
        AgentAction(
            thought="drag",
            action_type=ActionType.DRAG,
            target_id=1,
            drag_end_target_id=999,
        ),
    ],
)
def test_rejects_pixels_and_cells_outside_the_grid(tagged, bad):
    client = FakeClient(ok(bad), ok(click(5)))
    action = Planner(client=client).decide("g", tagged)

    assert action.target_id == 5
    assert len(client.messages.calls) == 2


def test_empty_answer_is_retried(tagged):
    client = FakeClient(ok(None), ok(click(5)))

    assert Planner(client=client).decide("g", tagged).target_id == 5


def test_refusal_fails_without_retry(tagged):
    client = FakeClient(FakeResponse(None, stop_reason="refusal"))
    action = Planner(client=client).decide("g", tagged)

    assert action.action_type is ActionType.FAIL
    assert "refus" in action.thought.lower()
    assert len(client.messages.calls) == 1


# -------------------------------------------------------------------- refine


def test_refine_uses_the_zoom_prompt_and_minimal_schema(tagged, zoom_tagged):
    client = FakeClient(ok(ZoomChoice(thought="the button", target_id=19)))
    cell = Planner(client=client).refine("g", click(24), zoom_tagged)
    (call,) = client.messages.calls

    assert cell == 19
    assert call["output_format"] is ZoomChoice
    assert call["system"][0]["text"] == ZOOM_SYSTEM_PROMPT
    assert "8 columns x 8 rows" in user_text(call)
    assert np.array_equal(user_image(call), zoom_tagged.image)


def test_refine_names_the_drag_end(zoom_tagged):
    client = FakeClient(ok(ZoomChoice(thought="t", target_id=1)))
    drag = AgentAction(
        thought="Move", action_type=ActionType.DRAG, target_id=4, drag_end_target_id=9
    )
    Planner(client=client).refine("g", drag, zoom_tagged, drag_end=True)

    assert "end" in user_text(client.messages.calls[0]).lower()


def test_refine_retries_a_cell_outside_the_grid(zoom_tagged):
    client = FakeClient(
        ok(ZoomChoice(thought="t", target_id=64)),
        ok(ZoomChoice(thought="t", target_id=7)),
    )

    assert Planner(client=client).refine("g", click(), zoom_tagged) == 7


def test_refine_gives_up_with_none(zoom_tagged):
    client = FakeClient(validation_error(), validation_error())

    assert Planner(client=client).refine("g", click(), zoom_tagged) is None


def test_refine_refusal_returns_none(zoom_tagged):
    client = FakeClient(FakeResponse(None, stop_reason="refusal"))

    assert Planner(client=client).refine("g", click(), zoom_tagged) is None


# --------------------------------------------------------------------- usage


def test_usage_of_every_call_is_recorded(tagged):
    client = FakeClient(validation_error(), ok(click()))
    planner = Planner(client=client)
    planner.decide("g", tagged)

    # The failed call raised before returning usage; only the answered one counts.
    assert [u.input_tokens for u in planner.usage] == [2400]


# --------------------------------------------------------------- integration


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no API key")
def test_haiku_clicks_the_save_button_on_a_synthetic_screen():
    planner = Planner()
    goal = "Click the blue 'Guardar' (Save) button in the toolbar."
    tagged = VisualTagger().apply_labeled_grid(scenes.notes_app_frame())

    action = planner.decide(goal, tagged)
    assert action.action_type is ActionType.CLICK, action.thought

    x0, y0, x1, y1 = tagged.cell_bounds(action.target_id)
    sx, sy = tagged.frame.to_screen(x0, y0)
    ex, ey = tagged.frame.to_screen(x1 - 1, y1 - 1)
    region = scenes.region_frame(
        max(0, sx - 96), max(0, sy - 90), (ex - sx) + 192, (ey - sy) + 180
    )
    fine = VisualTagger().apply_labeled_grid(region, grid_cols=8, grid_rows=8)
    cell = planner.refine(goal, action, fine)

    assert cell is not None
    assert scenes.inside_save_button(*fine.cell_to_screen(cell))
