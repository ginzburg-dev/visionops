import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.fixtures import write_fixture_onnx
from visionops.api import create_app
from visionops.benchmark import benchmark_model
from visionops.inference import OnnxPersonDetector

pytest.importorskip("onnx")

INPUT_SIZE = 64


def _predictions():
    values = np.zeros((84, 2), dtype=np.float32)
    values[:4, 0] = (32.0, 32.0, 16.0, 32.0)
    values[4, 0] = 0.9
    values[:4, 1] = (10.0, 10.0, 4.0, 4.0)
    values[4, 1] = 0.1
    return values


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("onnx") / "fixture.onnx"
    return write_fixture_onnx(path, _predictions(), INPUT_SIZE)


def test_real_session_decodes_person_detections(model):
    detector = OnnxPersonDetector(model, confidence=0.25, iou=0.7)
    image = np.full((INPUT_SIZE, INPUT_SIZE, 3), 120, dtype=np.uint8)

    detections = detector.detect(image)

    assert detector.input_size == INPUT_SIZE
    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.9)
    assert detections[0].bbox_xyxy_px == pytest.approx((24.0, 16.0, 40.0, 48.0))
    assert detector.model_sha256 != "0" * 64


def test_api_serves_the_real_session(model):
    import cv2

    ok, encoded = cv2.imencode(".png", np.full((INPUT_SIZE, INPUT_SIZE, 3), 80, dtype=np.uint8))
    assert ok
    with TestClient(create_app(model)) as client:
        health = client.get("/health")
        predict = client.post(
            "/predict", files={"file": ("frame.png", encoded.tobytes(), "image/png")}
        )

    assert health.status_code == 200
    assert health.json()["input_size"] == INPUT_SIZE
    assert predict.status_code == 200
    assert predict.json()["person_count"] == 1


def test_benchmark_measures_the_real_session(model):
    detector = OnnxPersonDetector(model)
    report = benchmark_model(detector, [np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8)], 2, 5)

    assert report["samples"] == 5
    assert report["latency_ms"]["mean"] > 0
    assert report["providers"] == ["CPUExecutionProvider"]
    assert report["model_sha256"] == detector.model_sha256


def test_cli_prelabels_with_an_onnx_model(tmp_path, model):
    from tests.fixtures import write_image
    from visionops.cli import main
    from visionops.jsonl import read_jsonl
    from visionops.schemas import PredictionRecord

    write_image(tmp_path / "frames" / "a.png", 90)
    manifest = tmp_path / "ds" / "manifest.jsonl"
    assert main(["ingest", "--input", str(tmp_path / "frames"), "--out", str(tmp_path / "ds")]) == 0
    out = tmp_path / "ds" / "predictions.jsonl"
    assert (
        main(
            [
                "prelabel",
                "--manifest",
                str(manifest),
                "--model",
                str(model),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    (record,) = read_jsonl(out, PredictionRecord)
    assert record.person_count == 1
    assert record.model_id == "fixture.onnx"
