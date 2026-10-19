"""Tests for vision/som_tagger.py.

Frames are built from synthetic images, so nothing touches the real screen.
Drawing tests use a uniform mid-gray image: any pixel that differs from that
gray was painted by the tagger.
"""

import dataclasses

import numpy as np
import pytest

from vision.capture import Frame
from vision.som_tagger import TaggedFrame, VisualTagger

GRAY = 128


def _frame(width=1280, height=720, scale=1.0, origin=(0, 0)) -> Frame:
    image = np.full((height, width, 3), GRAY, dtype=np.uint8)
    original = (round(width / scale), round(height / scale))
    return Frame(image=image, original_size=original, scale=scale, origin=origin)


def _painted(tagged: TaggedFrame, x: int, y: int) -> bool:
    return bool(np.any(tagged.image[y, x] != GRAY))


# --------------------------------------------------------------- grid geometry


def test_default_grid_is_20_by_12():
    tagged = VisualTagger().apply_labeled_grid(_frame())

    assert (tagged.grid_cols, tagged.grid_rows) == (20, 12)
    assert tagged.cell_count == 240


def test_cell_ids_are_row_major_from_zero():
    tagged = VisualTagger().apply_labeled_grid(_frame())  # 64x60 px cells

    assert tagged.cell_bounds(0) == (0, 0, 64, 60)
    assert tagged.cell_bounds(1) == (64, 0, 128, 60)
    assert tagged.cell_bounds(20) == (0, 60, 64, 120)  # first cell of row 2
    assert tagged.cell_bounds(239) == (1216, 660, 1280, 720)


def test_custom_grid_size():
    tagged = VisualTagger().apply_labeled_grid(_frame(), grid_cols=4, grid_rows=2)

    assert tagged.cell_count == 8
    assert tagged.cell_bounds(5) == (320, 360, 640, 720)


def test_uneven_division_covers_the_image_without_gaps():
    tagged = VisualTagger().apply_labeled_grid(
        _frame(width=100, height=50), grid_cols=3, grid_rows=2
    )
    bounds = [tagged.cell_bounds(i) for i in range(tagged.cell_count)]

    assert [b[0] for b in bounds[:3]] == [0, 33, 67]
    assert [b[2] for b in bounds[:3]] == [33, 67, 100]
    assert [b[1] for b in bounds[::3]] == [0, 25]
    assert [b[3] for b in bounds[::3]] == [25, 50]


def test_cell_center_is_inside_the_cell():
    tagged = VisualTagger().apply_labeled_grid(_frame())

    assert tagged.cell_center(0) == (32, 30)
    assert tagged.cell_center(21) == (96, 90)


@pytest.mark.parametrize("cell_id", [-1, 240, 1000])
def test_invalid_cell_id_raises(cell_id):
    tagged = VisualTagger().apply_labeled_grid(_frame())

    with pytest.raises(ValueError):
        tagged.cell_bounds(cell_id)
    with pytest.raises(ValueError):
        tagged.cell_to_screen(cell_id)


@pytest.mark.parametrize(
    ("cols", "rows"),
    [(0, 12), (20, 0), (-1, 12), (1281, 12), (20, 721)],  # 1281 > image width
)
def test_invalid_grid_size_raises(cols, rows):
    with pytest.raises(ValueError):
        VisualTagger().apply_labeled_grid(_frame(), grid_cols=cols, grid_rows=rows)


# ------------------------------------------------------- mapping to the screen


def test_cell_to_screen_on_full_resolution_frame():
    tagged = VisualTagger().apply_labeled_grid(_frame())

    assert tagged.cell_to_screen(0) == (32, 30)


def test_cell_to_screen_undoes_downscale():
    # 2560x1440 screen captured at half size
    tagged = VisualTagger().apply_labeled_grid(_frame(scale=0.5))

    assert tagged.cell_to_screen(0) == (64, 60)


def test_cell_to_screen_adds_region_origin():
    # A region (e.g. a browser window) starting at (600, 200); the base of the
    # two-step zoom: tag a region, then map the chosen cell back to the screen.
    tagged = VisualTagger().apply_labeled_grid(
        _frame(width=400, height=300, origin=(600, 200)), grid_cols=4, grid_rows=3
    )

    assert tagged.cell_center(0) == (50, 50)
    assert tagged.cell_to_screen(0) == (650, 250)


# --------------------------------------------------------------------- drawing


def test_output_keeps_shape_and_does_not_modify_the_input():
    frame = _frame()
    tagged = VisualTagger().apply_labeled_grid(frame)

    assert tagged.image.shape == frame.image.shape
    assert tagged.image.dtype == np.uint8
    assert np.all(frame.image == GRAY)
    assert tagged.frame is frame


def test_draws_interior_grid_lines():
    tagged = VisualTagger().apply_labeled_grid(_frame())

    assert _painted(tagged, 64, 300)  # vertical line between columns 0 and 1
    assert _painted(tagged, 640, 60)  # horizontal line between rows 0 and 1


def test_cell_centers_stay_clean():
    tagged = VisualTagger().apply_labeled_grid(_frame())

    for cell_id in (0, 21, 239):
        assert not _painted(tagged, *tagged.cell_center(cell_id))


def test_draws_a_label_in_the_top_left_corner_of_every_cell():
    tagged = VisualTagger().apply_labeled_grid(_frame())

    for cell_id in range(tagged.cell_count):
        x0, y0, _, _ = tagged.cell_bounds(cell_id)
        assert _painted(tagged, x0 + 3, y0 + 3), f"no label in cell {cell_id}"


def test_labels_show_different_ids():
    tagged = VisualTagger().apply_labeled_grid(_frame())

    def label_patch(cell_id):
        x0, y0, _, _ = tagged.cell_bounds(cell_id)
        return tagged.image[y0 + 2 : y0 + 20, x0 + 2 : x0 + 20]

    assert not np.array_equal(label_patch(1), label_patch(2))


def test_higher_alpha_makes_lines_stronger():
    frame = _frame()
    faint = VisualTagger(alpha=0.2).apply_labeled_grid(frame)
    strong = VisualTagger(alpha=0.9).apply_labeled_grid(frame)

    def distance_from_gray(tagged):
        return np.abs(tagged.image[300, 64].astype(int) - GRAY).sum()

    assert distance_from_gray(strong) > distance_from_gray(faint) > 0


@pytest.mark.parametrize("alpha", [0, -0.1, 1.5])
def test_invalid_alpha_raises(alpha):
    with pytest.raises(ValueError):
        VisualTagger(alpha=alpha)


def test_default_alpha_is_half():
    assert VisualTagger().alpha == 0.5


# ------------------------------------------------------------------ TaggedFrame


def test_tagged_frame_is_frozen():
    tagged = VisualTagger().apply_labeled_grid(_frame())

    with pytest.raises(dataclasses.FrozenInstanceError):
        tagged.grid_cols = 5
