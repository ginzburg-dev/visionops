import json

import pytest

from tests.fixtures import make_prediction
from visionops.annotations import export_labelstudio
from visionops.jsonl import write_jsonl
from visionops.schemas import SelectionRecord


def _selection(tmp_path, predictions):
    records = [
        SelectionRecord(
            prediction=prediction,
            rank=index + 1,
            strategy="uncertainty",
            uncertainty_score=0.5,
            diversity_score=0.25,
        )
        for index, prediction in enumerate(predictions)
    ]
    return write_jsonl(tmp_path / "selection.jsonl", records)


def test_rectangles_use_percentages_and_model_metadata(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "frame-0.jpg").write_bytes(b"not-decoded-here")
    path = _selection(tmp_path, [make_prediction("frame-0", (0.8,))])

    out = export_labelstudio(path, tmp_path / "tasks.json")
    tasks = json.loads(out.read_text(encoding="utf-8"))

    (task,) = tasks
    (result,) = task["predictions"][0]["result"]
    assert result["type"] == "rectanglelabels"
    assert result["value"]["rectanglelabels"] == ["person"]
    assert result["value"]["x"] == pytest.approx(100.0 * 1.0 / 32.0)
    assert result["value"]["height"] == pytest.approx(100.0 * 18.0 / 24.0)
    assert task["meta"]["model_sha256"] == "b" * 64
    assert task["meta"]["review_required"] is True
    assert task["predictions"][0]["model_version"] == "fake-model"


def test_empty_predictions_export_as_valid_tasks(tmp_path):
    path = _selection(tmp_path, [make_prediction("frame-1", ())])
    out = export_labelstudio(path, tmp_path / "tasks.json")
    (task,) = json.loads(out.read_text(encoding="utf-8"))
    assert task["predictions"][0]["result"] == []
    assert task["meta"]["person_count"] == 0


def test_source_image_reference_resolves(tmp_path):
    from pathlib import Path

    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "frame-0.jpg").write_bytes(b"x")
    path = _selection(tmp_path, [make_prediction("frame-0", (0.8,))])
    (task,) = json.loads(export_labelstudio(path, tmp_path / "tasks.json").read_text("utf-8"))
    assert Path(task["data"]["image"]).is_file()
