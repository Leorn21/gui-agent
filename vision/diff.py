"""Frame difference detection for the GUI agent.

Tells whether an action changed the screen and where. Two frames of the same
area are compared pixel by pixel; nearby changed pixels are merged into
regions (so typed letters count as one change) and regions with fewer changed
pixels than a minimum area are dropped (so a blinking caret does not count).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from vision.capture import Frame

PIXEL_THRESHOLD = 10  # a pixel changed if some channel moved by more than this
DEFAULT_MIN_AREA = 0.0005  # share of the image a region must change (0.05%)
MERGE_RADIUS = 4  # px; changes closer than ~2x this merge into one region
DEFAULT_TIMEOUT = 2.0
DEFAULT_INTERVAL = 0.1


@dataclass(frozen=True)
class Region:
    """A rectangle in absolute screen pixels."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, eq=False)
class DiffResult:
    """Outcome of comparing two frames.

    Attributes:
        changed: Whether any region passed the minimum area.
        changed_ratio: Share of the image's pixels that changed in those regions.
        regions: Changed regions in screen pixels, most changed pixels first.
        bounds: Rectangle enclosing all regions, or ``None`` when unchanged.
    """

    changed: bool
    changed_ratio: float
    regions: tuple[Region, ...]
    bounds: Region | None


_UNCHANGED = DiffResult(changed=False, changed_ratio=0.0, regions=(), bounds=None)


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


class FrameDiff:
    """Compares frames and waits for the screen to change."""

    def __init__(self, min_area: float = DEFAULT_MIN_AREA) -> None:
        """
        Args:
            min_area: Minimum share of the image, in ``[0, 1)``, a region must
                change to count.

        Raises:
            ValueError: If ``min_area`` is out of range.
        """

        if not 0 <= min_area < 1:
            raise ValueError(f"min_area must be in [0, 1), got {min_area}")
        self.min_area = min_area

    def compare(self, before: Frame, after: Frame) -> DiffResult:
        """Compare two frames of the same screen area.

        Raises:
            ValueError: If the frames differ in size, origin or scale.
        """

        if (
            before.image.shape != after.image.shape
            or before.origin != after.origin
            or before.scale != after.scale
        ):
            raise ValueError("Frames must cover the same area at the same scale")

        delta = cv2.absdiff(before.image, after.image).max(axis=2)
        mask = delta > PIXEL_THRESHOLD
        if not mask.any():
            return _UNCHANGED

        size = 2 * MERGE_RADIUS + 1
        grown = cv2.dilate(mask.astype(np.uint8), np.ones((size, size), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            grown, connectivity=8
        )
        changed_labels = np.where(mask, labels, 0)
        pixels = np.bincount(changed_labels.ravel(), minlength=count)

        total = mask.size
        min_pixels = self.min_area * total
        kept = []  # (changed pixels, image bbox x0, y0, x1, y1)
        for label in range(1, count):
            if pixels[label] < min_pixels:
                continue
            bx, by, bw, bh = stats[label, :4]
            ys, xs = np.nonzero(changed_labels[by : by + bh, bx : bx + bw] == label)
            kept.append(
                (
                    int(pixels[label]),
                    bx + int(xs.min()),
                    by + int(ys.min()),
                    bx + int(xs.max()) + 1,
                    by + int(ys.max()) + 1,
                )
            )

        if not kept:
            return _UNCHANGED

        kept.sort(key=lambda item: item[0], reverse=True)
        regions = tuple(self._to_screen(after, *box) for _, *box in kept)
        x0 = min(r.x for r in regions)
        y0 = min(r.y for r in regions)
        x1 = max(r.x + r.width for r in regions)
        y1 = max(r.y + r.height for r in regions)
        return DiffResult(
            changed=True,
            changed_ratio=sum(item[0] for item in kept) / total,
            regions=regions,
            bounds=Region(x=x0, y=y0, width=x1 - x0, height=y1 - y0),
        )

    def wait_for_change(
        self,
        before: Frame,
        capture: Callable[[], Frame],
        timeout: float = DEFAULT_TIMEOUT,
        interval: float = DEFAULT_INTERVAL,
        settle: bool = True,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> tuple[DiffResult, Frame]:
        """Capture until the screen differs from ``before``, then until it settles.

        Detecting the change and settling share one ``timeout``. "Settled"
        means two consecutive captures show no change.

        Args:
            before: Frame taken before the action.
            capture: Returns a new frame of the same area.
            timeout: Seconds to keep capturing.
            interval: Seconds between captures.
            settle: Also wait for the screen to stop changing.

        Returns:
            The comparison against ``before`` and the last captured frame.

        Raises:
            ValueError: If ``timeout`` is negative or ``interval`` not positive.
        """

        if timeout < 0:
            raise ValueError(f"timeout must not be negative, got {timeout}")
        if interval <= 0:
            raise ValueError(f"interval must be positive, got {interval}")

        polls_left = math.ceil(timeout / interval - 1e-9)
        deadline = clock() + timeout

        def next_frame() -> Frame | None:
            nonlocal polls_left
            if polls_left <= 0 or clock() >= deadline:
                return None
            sleep(interval)
            polls_left -= 1
            return capture()

        current = capture()
        result = self.compare(before, current)
        while not result.changed:
            following = next_frame()
            if following is None:
                return result, current
            current = following
            result = self.compare(before, current)

        if not settle:
            return result, current

        while True:
            following = next_frame()
            if following is None:
                return self.compare(before, current), current
            settled = not self.compare(current, following).changed
            current = following
            if settled:
                return self.compare(before, current), current

    @staticmethod
    def _to_screen(frame: Frame, x0: int, y0: int, x1: int, y1: int) -> Region:
        ox, oy = frame.origin
        left = ox + _round_half_up(x0 / frame.scale)
        top = oy + _round_half_up(y0 / frame.scale)
        right = ox + _round_half_up(x1 / frame.scale)
        bottom = oy + _round_half_up(y1 / frame.scale)
        return Region(x=left, y=top, width=right - left, height=bottom - top)


__all__ = [
    "DEFAULT_INTERVAL",
    "DEFAULT_MIN_AREA",
    "DEFAULT_TIMEOUT",
    "PIXEL_THRESHOLD",
    "DiffResult",
    "FrameDiff",
    "Region",
]
