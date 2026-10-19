"""VLM planner: asks Claude which action to take next.

Each call sends one tagged screenshot (PNG, lossless) plus a text prompt, and
uses structured outputs so the SDK returns an already validated Pydantic
object. Answers that use pixel coordinates or cells outside the grid are
rejected; an invalid answer is retried once with the reason, then the step
fails. Clicks are refined in a second call on an enlarged region
(``refine``), which returns only a finer cell.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import anthropic
import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.prompts import (
    DEFAULT_HISTORY_LIMIT,
    SYSTEM_PROMPT,
    ZOOM_SYSTEM_PROMPT,
    StepRecord,
    build_step_prompt,
    build_zoom_prompt,
    format_history,
)
from schemas.actions import ActionType, AgentAction
from vision.som_tagger import TaggedFrame

DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_EFFORT = "low"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_RETRIES = 1
EFFORTS = ("low", "medium", "high", "xhigh", "max")

T = TypeVar("T", bound=BaseModel)


class ZoomChoice(BaseModel):
    """Answer of the zoom step: only a finer cell."""

    model_config = ConfigDict(extra="forbid")

    thought: str = Field(description="Why this cell contains the target")
    target_id: int = Field(ge=0, description="Cell of the finer grid")


def _encode_png(image: NDArray[np.uint8]) -> str:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode the screenshot as PNG")
    return base64.standard_b64encode(buffer.tobytes()).decode("ascii")


def _validation_summary(exc: ValidationError) -> str:
    return "; ".join(error["msg"] for error in exc.errors())[:300]


class Planner:
    """Chooses actions with Claude from tagged screenshots."""

    def __init__(
        self,
        client: Any = None,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """
        Args:
            client: An ``anthropic.Anthropic`` client; one is created from the
                environment (``ANTHROPIC_API_KEY``) when omitted.
            model: Claude model ID.
            effort: How much the model reasons: low, medium, high, xhigh, max.
            max_tokens: Output token cap per call.
            history_limit: Previous steps included in each prompt.
            max_retries: Extra attempts after an invalid answer.

        Raises:
            ValueError: On an unknown effort, ``history_limit < 1`` or
                negative ``max_retries``.
        """

        if effort not in EFFORTS:
            raise ValueError(f"effort must be one of {EFFORTS}, got {effort!r}")
        if history_limit < 1:
            raise ValueError(f"history_limit must be at least 1, got {history_limit}")
        if max_retries < 0:
            raise ValueError(f"max_retries must not be negative, got {max_retries}")

        self._client = client if client is not None else anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.history_limit = history_limit
        self.max_retries = max_retries
        self.usage: list[Any] = []  # response.usage of every answered call

    def decide(
        self,
        goal: str,
        tagged: TaggedFrame,
        history: Sequence[StepRecord] = (),
    ) -> AgentAction:
        """Choose the next action on ``tagged``.

        Never raises for a bad model answer: returns a ``FAIL`` action instead.
        """

        prompt = build_step_prompt(
            goal,
            format_history(history, self.history_limit),
            tagged.grid_cols,
            tagged.grid_rows,
        )

        def check(action: AgentAction) -> str | None:
            if action.coordinate is not None or action.drag_end_coordinate is not None:
                return "use grid cells (target_id), never pixel coordinates."
            for name in ("target_id", "drag_end_target_id"):
                cell = getattr(action, name)
                if cell is not None and cell >= tagged.cell_count:
                    return (
                        f"{name} {cell} is not on the grid "
                        f"(cells 0 to {tagged.cell_count - 1})."
                    )
            return None

        action, failure = self._ask(
            SYSTEM_PROMPT, prompt, tagged.image, AgentAction, check
        )
        if action is not None:
            return action
        return AgentAction(thought=f"Planner: {failure}", action_type=ActionType.FAIL)

    def refine(
        self,
        goal: str,
        action: AgentAction,
        zoom_tagged: TaggedFrame,
        drag_end: bool = False,
    ) -> int | None:
        """Choose the finer cell for ``action`` on an enlarged region.

        Returns:
            The cell of ``zoom_tagged``, or ``None`` if no valid answer was
            obtained (the caller can fall back to the coarse cell).
        """

        prompt = build_zoom_prompt(
            goal, action, zoom_tagged.grid_cols, zoom_tagged.grid_rows, drag_end
        )

        def check(choice: ZoomChoice) -> str | None:
            if choice.target_id >= zoom_tagged.cell_count:
                return (
                    f"target_id {choice.target_id} is not on the grid "
                    f"(cells 0 to {zoom_tagged.cell_count - 1})."
                )
            return None

        choice, _ = self._ask(
            ZOOM_SYSTEM_PROMPT, prompt, zoom_tagged.image, ZoomChoice, check
        )
        return None if choice is None else choice.target_id

    def _ask(
        self,
        system: str,
        prompt: str,
        image: NDArray[np.uint8],
        output_format: type[T],
        check: Callable[[T], str | None],
    ) -> tuple[T | None, str]:
        """Call the model until it gives a valid answer or retries run out.

        Returns:
            ``(answer, "")`` on success, ``(None, reason)`` otherwise.
        """

        image_block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": _encode_png(image),
            },
        }
        rejection = ""
        for _ in range(self.max_retries + 1):
            text = prompt
            if rejection:
                text += (
                    f"\n\nYour previous answer was rejected: {rejection} "
                    "Answer again following the rules."
                )
            try:
                response = self._client.messages.parse(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=[
                        {
                            "type": "text",
                            "text": system,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[
                        {
                            "role": "user",
                            "content": [image_block, {"type": "text", "text": text}],
                        }
                    ],
                    output_format=output_format,
                    output_config={"effort": self.effort},
                )
            except ValidationError as exc:
                rejection = f"invalid fields: {_validation_summary(exc)}."
                continue

            self.usage.append(response.usage)
            if response.stop_reason == "refusal":
                return None, "the model refused to choose an action."
            answer = response.parsed_output
            if answer is None:
                rejection = "the answer was empty."
                continue
            problem = check(answer)
            if problem:
                rejection = problem
                continue
            return answer, ""

        return (
            None,
            f"invalid answer after {self.max_retries + 1} attempts: {rejection}",
        )


__all__ = [
    "DEFAULT_EFFORT",
    "DEFAULT_HISTORY_LIMIT",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "EFFORTS",
    "Planner",
    "ZoomChoice",
]
