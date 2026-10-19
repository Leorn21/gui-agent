"""Set-of-Mark grid overlay for the GUI agent.

Draws a numbered grid over a captured ``Frame`` so the VLM can answer with a
cell ID (``AgentAction.target_id``) instead of raw coordinates. The returned
``TaggedFrame`` maps that ID back to absolute screen pixels through the
source frame, so the model never does coordinate arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.capture import Frame

DEFAULT_GRID_COLS = 20
DEFAULT_GRID_ROWS = 12
DEFAULT_ALPHA = 0.5

# Fixed high-contrast style (BGR).
LINE_COLOR = (255, 0, 255)  # magenta
LABEL_BACKGROUND = (0, 0, 0)
LABEL_TEXT_COLOR = (255, 255, 255)
LINE_THICKNESS = 1
LABEL_PADDING = 2
LABEL_OFFSET = 1  # gap between the cell corner and the label box
FONT = cv2.FONT_HERSHEY_SIMPLEX
MAX_FONT_SCALE = 0.5
MIN_FONT_SCALE = 0.25


def _edges(length: int, parts: int) -> list[int]:
    """Split ``length`` pixels into ``parts`` contiguous spans, rounding half up."""

    return [(2 * i * length + parts) // (2 * parts) for i in range(parts + 1)]


@dataclass(frozen=True, eq=False)
class TaggedFrame:
    """A frame with a numbered grid drawn on it.

    Attributes:
        image: Annotated BGR copy of ``frame.image``.
        frame: The source frame (used to map cells back to the screen).
        grid_cols: Number of grid columns.
        grid_rows: Number of grid rows.
    """

    image: NDArray[np.uint8]
    frame: Frame
    grid_cols: int
    grid_rows: int

    @property
    def cell_count(self) -> int:
        """Total number of cells; valid IDs are ``0 .. cell_count - 1``."""

        return self.grid_cols * self.grid_rows

    def cell_bounds(self, cell_id: int) -> tuple[int, int, int, int]:
        """Return ``(x0, y0, x1, y1)`` of a cell in image pixels (end-exclusive).

        IDs are row-major: 0 is top-left, ``grid_cols`` starts the second row.

        Raises:
            ValueError: If ``cell_id`` is out of range.
        """

        if not 0 <= cell_id < self.cell_count:
            raise ValueError(
                f"cell_id must be between 0 and {self.cell_count - 1}, got {cell_id}"
            )
        width, height = self.frame.size
        row, col = divmod(cell_id, self.grid_cols)
        xs = _edges(width, self.grid_cols)
        ys = _edges(height, self.grid_rows)
        return xs[col], ys[row], xs[col + 1], ys[row + 1]

    def cell_center(self, cell_id: int) -> tuple[int, int]:
        """Return the center pixel of a cell in image coordinates."""

        x0, y0, x1, y1 = self.cell_bounds(cell_id)
        return (x0 + x1) // 2, (y0 + y1) // 2

    def cell_to_screen(self, cell_id: int) -> tuple[int, int]:
        """Return the cell center as absolute screen pixels."""

        return self.frame.to_screen(*self.cell_center(cell_id))


class VisualTagger:
    """Draws Set-of-Mark grids over frames."""

    def __init__(self, alpha: float = DEFAULT_ALPHA) -> None:
        """
        Args:
            alpha: Opacity of grid lines and label backgrounds, in ``(0, 1]``.
                Label text is always fully opaque.

        Raises:
            ValueError: If ``alpha`` is out of range.
        """

        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha

    def apply_labeled_grid(
        self,
        frame: Frame,
        grid_cols: int = DEFAULT_GRID_COLS,
        grid_rows: int = DEFAULT_GRID_ROWS,
    ) -> TaggedFrame:
        """Return a copy of ``frame`` with a numbered grid drawn on it.

        Interior lines separate the cells and each cell gets its ID in the
        top-left corner, leaving the center (the click point) uncovered.

        Raises:
            ValueError: If the grid has no cells or cells narrower than 1 px.
        """

        width, height = frame.size
        if not 1 <= grid_cols <= width:
            raise ValueError(
                f"grid_cols must be between 1 and {width}, got {grid_cols}"
            )
        if not 1 <= grid_rows <= height:
            raise ValueError(
                f"grid_rows must be between 1 and {height}, got {grid_rows}"
            )

        xs = _edges(width, grid_cols)
        ys = _edges(height, grid_rows)
        labels = [str(i) for i in range(grid_cols * grid_rows)]
        font_scale = self._font_scale(
            labels[-1],
            min(b - a for a, b in zip(xs, xs[1:], strict=False)),
            min(b - a for a, b in zip(ys, ys[1:], strict=False)),
        )

        # Lines and label boxes are alpha-blended; text goes on top, opaque.
        overlay = frame.image.copy()
        for x in xs[1:-1]:
            cv2.line(overlay, (x, 0), (x, height - 1), LINE_COLOR, LINE_THICKNESS)
        for y in ys[1:-1]:
            cv2.line(overlay, (0, y), (width - 1, y), LINE_COLOR, LINE_THICKNESS)

        placements = []
        for cell_id, label in enumerate(labels):
            row, col = divmod(cell_id, grid_cols)
            (text_w, text_h), baseline = cv2.getTextSize(label, FONT, font_scale, 1)
            box_x0 = xs[col] + LABEL_OFFSET
            box_y0 = ys[row] + LABEL_OFFSET
            box_x1 = box_x0 + text_w + 2 * LABEL_PADDING
            box_y1 = box_y0 + text_h + baseline + 2 * LABEL_PADDING
            cv2.rectangle(
                overlay, (box_x0, box_y0), (box_x1, box_y1), LABEL_BACKGROUND, -1
            )
            placements.append(
                (label, (box_x0 + LABEL_PADDING, box_y0 + LABEL_PADDING + text_h))
            )

        image = cv2.addWeighted(overlay, self.alpha, frame.image, 1 - self.alpha, 0)
        for label, origin in placements:
            cv2.putText(
                image, label, origin, FONT, font_scale, LABEL_TEXT_COLOR, 1, cv2.LINE_AA
            )

        return TaggedFrame(
            image=image, frame=frame, grid_cols=grid_cols, grid_rows=grid_rows
        )

    @staticmethod
    def _font_scale(longest_label: str, cell_w: int, cell_h: int) -> float:
        """Largest font scale whose label box stays in the cell's top-left quarter."""

        max_w = cell_w // 2 - LABEL_OFFSET - 2 * LABEL_PADDING - 1
        max_h = cell_h // 2 - LABEL_OFFSET - 2 * LABEL_PADDING - 1
        scale = MAX_FONT_SCALE
        while scale > MIN_FONT_SCALE:
            (text_w, text_h), baseline = cv2.getTextSize(longest_label, FONT, scale, 1)
            if text_w <= max_w and text_h + baseline <= max_h:
                break
            scale -= 0.05
        return max(scale, MIN_FONT_SCALE)


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_GRID_COLS",
    "DEFAULT_GRID_ROWS",
    "TaggedFrame",
    "VisualTagger",
]
