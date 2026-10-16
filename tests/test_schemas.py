import json
from pathlib import Path

import pytest

from visionops.jsonl import read_jsonl, write_jsonl
from visionops.schemas import Detection, FrameRecord, PredictionRecord


def make_frame() -> FrameRecord:
    return FrameRecord(
        image_id="clip-a-000001",
        image_path="images/clip-a-000001.jpg",
        source_path="clip-a.mp4",
        source_id="clip-a",
        frame_index=1,
        timestamp_s=0.04,
        width=1920,
        height=1080,
        split_group="clip-a",
        content_sha256="a" * 64,
    )


def test_detection_rejects_non_finite_or_reversed_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        Detection(0, "person", 0.8, (0.0, 0.0, float("nan"), 10.0))
    with pytest.raises(ValueError, match="ordering"):
        Detection(0, "person", 0.8, (20.0, 0.0, 10.0, 10.0))
    with pytest.raises(ValueError, match="between 0 and 1"):
        Detection(0, "person", 1.1, (0.0, 0.0, 10.0, 10.0))


def test_prediction_count_must_equal_retained_people() -> None:
    detection = Detection(0, "person", 0.8, (0.0, 0.0, 10.0, 10.0))

    with pytest.raises(ValueError, match="person_count"):
        PredictionRecord(
            image=make_frame(),
            model_id="yolo11n",
            model_sha256="b" * 64,
            confidence_threshold=0.25,
            iou_threshold=0.45,
            detections=(detection,),
            person_count=0,
        )


def test_empty_prediction_round_trips_through_strict_jsonl(tmp_path: Path) -> None:
    record = PredictionRecord(
        image=make_frame(),
        model_id="yolo11n",
        model_sha256="b" * 64,
        confidence_threshold=0.25,
        iou_threshold=0.45,
        detections=(),
        person_count=0,
    )
    path = tmp_path / "predictions.jsonl"

    write_jsonl(path, [record])

    assert read_jsonl(path, PredictionRecord) == [record]
    assert json.loads(path.read_text(encoding="utf-8"))["person_count"] == 0


def test_jsonl_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    payload = make_frame().to_dict()
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        read_jsonl(path, FrameRecord)


def test_jsonl_rejects_non_finite_json_constant(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text('{"schema_version": 1, "timestamp_s": NaN}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite"):
        read_jsonl(path, FrameRecord)
