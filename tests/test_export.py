import json

import numpy as np
import pytest

from tests.fixtures import write_image
from visionops.export import ParityError, compare_detections, export_onnx
from visionops.schemas import Detection


def _detection(x1=10.0, confidence=0.9):
    return Detection(
        class_id=0, class_name="person", confidence=confidence, bbox_xyxy_px=(x1, 5.0, 30.0, 40.0)
    )


class _Detector:
    def __init__(self, detections, input_size=640):
        self._detections = detections
        self.input_size = input_size

    def detect(self, image_bgr):
        assert isinstance(image_bgr, np.ndarray)
        return list(self._detections)


def _exporter(tmp_path, name="weights.onnx"):
    def run(**kwargs):
        produced = tmp_path / name
        produced.write_bytes(b"onnx-bytes")
        run.kwargs = kwargs
        return produced

    return run


def test_requested_destination_is_honored(tmp_path):
    exporter = _exporter(tmp_path)
    destination = tmp_path / "artifacts" / "person.onnx"

    out = export_onnx(
        tmp_path / "model.pt",
        destination,
        imgsz=640,
        exporter=exporter,
        onnx_detector_factory=lambda path: _Detector([]),
    )

    assert out == destination
    assert destination.is_file()
    assert not (tmp_path / "weights.onnx").exists()
    assert exporter.kwargs["opset"] == 12
    assert exporter.kwargs["imgsz"] == 640
    assert exporter.kwargs["dynamic"] is False


def test_input_shape_is_validated(tmp_path):
    with pytest.raises(ValueError, match="expects input size 320, requested 640"):
        export_onnx(
            tmp_path / "model.pt",
            tmp_path / "out.onnx",
            imgsz=640,
            exporter=_exporter(tmp_path),
            onnx_detector_factory=lambda path: _Detector([], input_size=320),
        )


def test_matching_detections_pass_tolerance_and_are_recorded(tmp_path):
    write_image(tmp_path / "sample.png", 120)
    source = _Detector([_detection()])
    onnx = _Detector([_detection(x1=10.5, confidence=0.905)])

    out = export_onnx(
        tmp_path / "model.pt",
        tmp_path / "out.onnx",
        imgsz=640,
        sample_images=[tmp_path / "sample.png"],
        exporter=_exporter(tmp_path),
        source_detector=source,
        onnx_detector_factory=lambda path: onnx,
        dataset_config_sha256="d" * 64,
    )

    manifest = json.loads((out.with_name("model-manifest.json")).read_text(encoding="utf-8"))
    assert manifest["parity"]["verified"] is True
    assert manifest["parity"]["samples"][0]["max_box_delta_px"] == pytest.approx(0.5)
    assert manifest["preprocessing"]["input_size"] == 640
    assert manifest["class_map"] == {"0": "person"}
    assert manifest["dataset_config_sha256"] == "d" * 64
    assert manifest["opset"] == 12


def test_box_divergence_fails_with_diagnostics(tmp_path):
    write_image(tmp_path / "sample.png", 120)
    with pytest.raises(ParityError, match="diverged by"):
        export_onnx(
            tmp_path / "model.pt",
            tmp_path / "out.onnx",
            imgsz=640,
            sample_images=[tmp_path / "sample.png"],
            exporter=_exporter(tmp_path),
            source_detector=_Detector([_detection()]),
            onnx_detector_factory=lambda path: _Detector([_detection(x1=25.0)]),
        )


def test_detection_count_divergence_is_reported():
    with pytest.raises(ParityError, match="returned 0 person detections"):
        compare_detections([_detection()], [])


def test_confidence_divergence_is_reported():
    with pytest.raises(ParityError, match="confidence"):
        compare_detections([_detection()], [_detection(confidence=0.5)])
