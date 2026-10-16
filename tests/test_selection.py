import numpy as np
import pytest

from tests.fixtures import make_prediction
from visionops.selection import HASH_BITS, dhash, frame_uncertainty, select_records


def _predictions(count=6):
    return [make_prediction(f"frame-{index}", (0.3 + 0.1 * index,)) for index in range(count)]


def test_random_selection_is_repeatable_for_a_seed():
    predictions = _predictions()
    first = select_records(predictions, 3, "random", 7)
    second = select_records(predictions, 3, "random", 7)
    assert [item.prediction.image.image_id for item in first] == [
        item.prediction.image.image_id for item in second
    ]
    assert [item.rank for item in first] == [1, 2, 3]


def test_budget_is_clipped_to_available_frames():
    assert len(select_records(_predictions(2), 10, "random", 0)) == 2


def test_budget_must_be_positive():
    with pytest.raises(ValueError, match="budget must be greater than zero"):
        select_records(_predictions(2), 0, "random", 0)


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown selection strategy"):
        select_records(_predictions(2), 1, "active", 0)


def test_empty_scenes_are_selectable_and_scored_highest():
    empty = make_prediction("frame-9", ())
    assert frame_uncertainty(empty) == 0.75
    records = select_records([empty, make_prediction("frame-1", (1.0,))], 1, "uncertainty", 0)
    assert records[0].prediction.image.image_id == "frame-9"
    assert records[0].prediction.person_count == 0


def test_duplicate_frames_are_suppressed(tmp_path):
    import cv2

    base = np.random.default_rng(0).integers(0, 255, (24, 32, 3), dtype=np.uint8)
    (tmp_path / "images").mkdir()
    for index in range(3):
        image = base.copy() if index < 2 else base[::-1, ::-1].copy()
        assert cv2.imwrite(str(tmp_path / "images" / f"frame-{index}.jpg"), image)
    predictions = [make_prediction(f"frame-{index}", (0.6,)) for index in range(3)]

    records = select_records(predictions, 5, "random", 0, root=tmp_path)
    ids = sorted(item.prediction.image.image_id for item in records)
    assert ids == ["frame-0", "frame-2"]


def test_uncertainty_ordering_is_deterministic_and_scored():
    predictions = [
        make_prediction("frame-0", (0.95,)),
        make_prediction("frame-1", (0.52,)),
        make_prediction("frame-2", (0.90,)),
    ]
    records = select_records(predictions, 3, "uncertainty", 0)
    assert [item.prediction.image.image_id for item in records][0] == "frame-1"
    assert records[0].diversity_score == 1.0
    assert all(0.0 <= item.uncertainty_score <= 1.0 for item in records)
    assert records == select_records(predictions, 3, "uncertainty", 0)


def test_dhash_is_stable_and_sized(tmp_path):
    import cv2

    image = np.random.default_rng(1).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    path = tmp_path / "a.png"
    assert cv2.imwrite(str(path), image)
    value = dhash(path)
    assert value == dhash(path)
    assert 0 <= value < 2**HASH_BITS
