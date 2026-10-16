import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from visionops.api import create_app
from visionops.schemas import Detection


class _Detector:
    model_id = "person.onnx"
    model_sha256 = "e" * 64
    input_size = 640
    confidence = 0.25
    iou = 0.7

    def __init__(self, path):
        self.path = path

    def detect(self, image_bgr):
        return [
            Detection(
                class_id=0, class_name="person", confidence=0.88, bbox_xyxy_px=(1.0, 2.0, 9.0, 20.0)
            )
        ]


def _client(factory=_Detector, **kwargs):
    return TestClient(create_app("model.onnx", detector_factory=factory, **kwargs))


def _png_bytes(width=32, height=24):
    image = np.full((height, width, 3), 90, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_health_reports_model_metadata():
    with _client() as client:
        response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_sha256"] == "e" * 64
    assert payload["classes"] == ["person"]


def test_predict_returns_detections_and_count():
    with _client() as client:
        response = client.post("/predict", files={"file": ("frame.png", _png_bytes(), "image/png")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["person_count"] == 1
    assert payload["detections"][0]["class_name"] == "person"
    assert payload["image"] == {"width": 32, "height": 24}


def test_corrupt_image_returns_structured_400():
    with _client() as client:
        response = client.post(
            "/predict", files={"file": ("frame.png", b"not-an-image", "image/png")}
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "undecodable_image"


def test_unsupported_content_type_returns_415():
    with _client() as client:
        response = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_upload_over_the_limit_returns_413():
    with _client(max_upload_bytes=128) as client:
        response = client.post("/predict", files={"file": ("frame.png", b"0" * 500, "image/png")})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "upload_too_large"


def test_unavailable_model_returns_503_on_both_routes():
    def failing(path):
        raise FileNotFoundError(f"ONNX model not found: {path}")

    with _client(factory=failing) as client:
        health = client.get("/health")
        predict = client.post("/predict", files={"file": ("f.png", _png_bytes(), "image/png")})
    assert health.status_code == 503
    assert predict.status_code == 503
    assert predict.json()["error"]["code"] == "model_unavailable"
    assert "ONNX model not found" in predict.json()["error"]["message"]


def test_inference_failure_is_reported_as_400():
    class _Broken(_Detector):
        def detect(self, image_bgr):
            raise ValueError("unsupported ONNX output shape")

    with _client(factory=_Broken) as client:
        response = client.post("/predict", files={"file": ("f.png", _png_bytes(), "image/png")})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "inference_failed"


@pytest.mark.parametrize("route", ["/health", "/predict"])
def test_routes_exist(route):
    app = create_app("model.onnx", detector_factory=_Detector)
    assert any(getattr(item, "path", None) == route for item in app.routes)
