"""Per-run log of what the agent saw and did.

Each run gets ``<base_dir>/<YYYYmmdd-HHMMSS>/`` with one annotated screenshot
per step (``step_01.png``...), ``steps.jsonl`` (one JSON object per step) and
``summary.json``. Only the newest ``keep`` run folders are kept; folders whose
name does not look like a run are never touched.

The screenshots are real captures of the user's screen: keep ``runs/`` out of
version control.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from agent.state_machine import RunResult, StepEvent

DEFAULT_BASE_DIR = "runs"
DEFAULT_KEEP = 10
_RUN_NAME = re.compile(r"^\d{8}-\d{6}(-\d+)?$")


class RunRecorder:
    """Writes one folder per agent run."""

    def __init__(
        self,
        base_dir: str | Path = DEFAULT_BASE_DIR,
        keep: int = DEFAULT_KEEP,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        """
        Args:
            base_dir: Folder that holds the run folders.
            keep: How many run folders to keep, the new one included.
            clock: Time source for folder names (tests inject one).

        Raises:
            ValueError: If ``keep`` is less than 1.
        """

        if keep < 1:
            raise ValueError(f"keep must be at least 1, got {keep}")
        self.base_dir = Path(base_dir)
        self.keep = keep
        self._clock = clock
        self.run_dir: Path | None = None
        self._goal = ""

    def start(self, goal: str) -> Path:
        """Create the folder for a new run, removing the oldest ones."""

        self.base_dir.mkdir(parents=True, exist_ok=True)
        runs = sorted(
            p for p in self.base_dir.iterdir() if p.is_dir() and _RUN_NAME.match(p.name)
        )
        for old in runs[: max(0, len(runs) - (self.keep - 1))]:
            shutil.rmtree(old)

        name = self._clock().strftime("%Y%m%d-%H%M%S")
        run_dir = self.base_dir / name
        suffix = 2
        while run_dir.exists():
            run_dir = self.base_dir / f"{name}-{suffix}"
            suffix += 1
        run_dir.mkdir()

        self.run_dir = run_dir
        self._goal = goal
        return run_dir

    def record_step(self, event: StepEvent, image: NDArray[np.uint8]) -> None:
        """Save the step's annotated screenshot and append its JSON line."""

        run_dir = self._require_run()
        cv2.imwrite(str(run_dir / f"step_{event.number:02d}.png"), image)
        line = {
            "number": event.number,
            "action": event.action.model_dump(mode="json"),
            "point": event.point,
            "end_point": event.end_point,
            "screen_changed": event.screen_changed,
            "changed_area": event.changed_area,
            "cost_usd": event.cost_usd,
        }
        with open(run_dir / "steps.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")

    def finish(self, result: RunResult) -> None:
        """Write ``summary.json`` for the run."""

        run_dir = self._require_run()
        summary = {
            "goal": self._goal,
            "status": str(result.status),
            "reason": result.reason,
            "steps": len(result.steps),
            "cost_usd": result.cost_usd,
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _require_run(self) -> Path:
        if self.run_dir is None:
            raise RuntimeError("Call start() before recording")
        return self.run_dir


__all__ = ["DEFAULT_BASE_DIR", "DEFAULT_KEEP", "RunRecorder"]
