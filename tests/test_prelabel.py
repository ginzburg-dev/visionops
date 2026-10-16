from pathlib import Path

import numpy as np
import pytest

from tests.fixtures import write_image
from visionops.ingest import ingest
from visionops.jsonl import read_jsonl
from visionops.prelabel import UltralyticsPersonDetector, prelabel_manifest
from visionops.schemas import PredictionRecord

MODEL_SHA = "a" * 64


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)


class _Result:
    names = {0: "person", 1: "bicycle"}

    def __init__(self, boxes):
        self.boxes = boxes


def _detector(boxes, **kwargs):
    return UltralyticsPersonDetector(
        Path("missing-weights.pt"), predictor=lambda image: [_Result(boxes)], **kwargs
    )


def test_keeps_only_person_class():
    detector = _detector(_Boxes([[1, 1, 5, 5], [2, 2, 6, 6]], [0.9, 0.9], [0, 1]))
    detections = detector.detect(np.zeros((24, 32, 3), dtype=np.uint8))
    assert [item.class_name for item in detections] == ["person"]


def test_enforces_confidence_threshold():
    detector = _detector(_Boxes([[1, 1, 5, 5], [2, 2, 6, 6]], [0.9, 0.2], [0, 0]), confidence=0.5)
    assert len(detector.detect(np.zeros((24, 32, 3), dtype=np.uint8))) == 1


def test_accepts_no_detections():
    detector = _detector(_Boxes(np.zeros((0, 4)), [], []))
    assert detector.detect(np.zeros((24, 32, 3), dtype=np.uint8)) == []


def test_clips_boxes_to_image_bounds():
    detector = _detector(_Boxes([[-10, -5, 500, 400]], [0.9], [0]))
    (detection,) = detector.detect(np.zeros((24, 32, 3), dtype=np.uint8))
    assert detection.bbox_xyxy_px == (0.0, 0.0, 32.0, 24.0)


def test_rejects_missing_local_weight_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="local model weight not found"):
        UltralyticsPersonDetector(tmp_path / "absent.pt")


def test_prelabel_preserves_manifest_order_and_counts(tmp_path):
    for index in range(3):
        write_image(tmp_path / "frames" / f"frame-{index}.png", index * 10)
    manifest = ingest([tmp_path / "frames"], tmp_path / "out", 1, None, 0)
    detector = _detector(_Boxes([[1, 1, 5, 5], [2, 2, 7, 7]], [0.9, 0.8], [0, 0]))
    out = prelabel_manifest(manifest, detector, tmp_path / "pred.jsonl", "fake-model", MODEL_SHA)

    records = read_jsonl(out, PredictionRecord)
    image_ids = [record.image.image_id for record in records]
    assert image_ids == sorted(image_ids)
    assert len(image_ids) == 3
    assert all(record.person_count == 2 for record in records)
    assert all(record.model_sha256 == MODEL_SHA for record in records)


def test_prelabel_records_empty_scenes(tmp_path):
    write_image(tmp_path / "frames" / "frame.png", 5)
    manifest = ingest([tmp_path / "frames"], tmp_path / "out", 1, None, 0)
    detector = _detector(_Boxes(np.zeros((0, 4)), [], []))
    out = prelabel_manifest(manifest, detector, tmp_path / "pred.jsonl", "fake-model", MODEL_SHA)

    (record,) = read_jsonl(out, PredictionRecord)
    assert record.person_count == 0
    assert record.warnings == ("no person detections retained",)


def test_predict_uses_square_letterbox_matching_the_onnx_input():
    captured: dict[str, object] = {}

    class _Model:
        def predict(self, **kwargs):
            captured.update(kwargs)
            return [_Result(_Boxes([[1, 1, 5, 5]], [0.9], [0]))]

    detector = UltralyticsPersonDetector(Path("missing-weights.pt"), predictor=lambda _: None)
    detector._predictor = None
    detector._model = _Model()
    detector.detect(np.zeros((24, 32, 3), dtype=np.uint8))

    assert captured["rect"] is False
    assert captured["imgsz"] == 640
