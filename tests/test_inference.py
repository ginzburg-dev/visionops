import cv2
import numpy as np
import pytest

from visionops.inference import (
    OnnxOutputLayoutError,
    OnnxPersonDetector,
    decode_yolo_output,
    letterbox,
)

ATTRIBUTES = 84


def _raw(boxes_scores):
    """Build a [1, 84, N] YOLO output from (cx, cy, w, h, person_score) rows."""
    predictions = np.zeros((ATTRIBUTES, len(boxes_scores)), dtype=np.float32)
    for index, (cx, cy, w, h, score) in enumerate(boxes_scores):
        predictions[:4, index] = (cx, cy, w, h)
        predictions[4, index] = score
    return predictions[None, ...]


def test_letterbox_scales_pads_and_converts_to_rgb():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    result = letterbox(image, 640)

    assert result.tensor.shape == (1, 3, 640, 640)
    assert result.scale == pytest.approx(3.2)
    assert result.pad_x == 0.0
    assert result.pad_y == pytest.approx(160.0)
    assert result.tensor[0, 0, 320, 320] == pytest.approx(1.0)
    assert result.tensor[0, 2, 320, 320] == pytest.approx(0.0)
    assert result.tensor[0, 0, 0, 0] == pytest.approx(114 / 255, abs=1e-6)


def test_letterbox_downscales_bilinearly_like_ultralytics():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
    expected = cv2.resize(image, (640, 360), interpolation=cv2.INTER_LINEAR)

    tensor = letterbox(image, 640).tensor[0].transpose(1, 2, 0)[140:500]
    restored = cv2.cvtColor((tensor * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)

    assert np.array_equal(restored, expected)


def test_letterbox_rejects_empty_images():
    with pytest.raises(ValueError, match="non-empty"):
        letterbox(np.zeros((0, 0, 3), dtype=np.uint8), 640)


def test_decode_restores_original_coordinates_and_clips():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    metadata = letterbox(image, 640)
    output = _raw([(320.0, 320.0, 640.0, 320.0, 0.9)])

    (detection,) = decode_yolo_output(output, metadata, 0.25, 0.7)

    assert detection.class_name == "person"
    assert detection.bbox_xyxy_px[0] == pytest.approx(0.0)
    assert detection.bbox_xyxy_px[2] == pytest.approx(200.0)
    assert detection.bbox_xyxy_px[1] == pytest.approx(0.0)
    assert detection.bbox_xyxy_px[3] == pytest.approx(100.0)


def test_decode_applies_confidence_and_nms():
    metadata = letterbox(np.zeros((640, 640, 3), dtype=np.uint8), 640)
    output = _raw(
        [
            (100.0, 100.0, 50.0, 50.0, 0.9),
            (102.0, 101.0, 50.0, 50.0, 0.8),
            (400.0, 400.0, 50.0, 50.0, 0.6),
            (500.0, 500.0, 50.0, 50.0, 0.1),
        ]
    )

    detections = decode_yolo_output(output, metadata, 0.25, 0.5)

    assert [round(item.confidence, 3) for item in detections] == [0.9, 0.6]


def test_decode_handles_empty_output():
    metadata = letterbox(np.zeros((64, 64, 3), dtype=np.uint8), 64)
    assert decode_yolo_output(_raw([]), metadata, 0.25, 0.7) == []


def test_both_output_layouts_decode_equivalently():
    metadata = letterbox(np.zeros((480, 640, 3), dtype=np.uint8), 640)
    channels_first = _raw([(300.0, 300.0, 40.0, 60.0, 0.77)])
    channels_last = channels_first.transpose(0, 2, 1)

    assert decode_yolo_output(channels_first, metadata, 0.25, 0.7) == decode_yolo_output(
        channels_last, metadata, 0.25, 0.7
    )


def test_unknown_layout_raises_shape_specific_error():
    metadata = letterbox(np.zeros((64, 64, 3), dtype=np.uint8), 64)
    with pytest.raises(OnnxOutputLayoutError, match=r"\(2, 2, 2, 2\)"):
        decode_yolo_output(np.zeros((2, 2, 2, 2), dtype=np.float32), metadata, 0.25, 0.7)


def test_missing_person_column_raises():
    metadata = letterbox(np.zeros((64, 64, 3), dtype=np.uint8), 64)
    with pytest.raises(OnnxOutputLayoutError, match="has no class 3"):
        decode_yolo_output(np.zeros((1, 10, 5), dtype=np.float32), metadata, 0.25, 0.7, 3)


class _FakeInput:
    name = "images"
    shape = [1, 3, 320, 320]


class _FakeSession:
    def __init__(self, output):
        self.output = output
        self.seen = None

    def get_inputs(self):
        return [_FakeInput()]

    def run(self, names, feed):
        self.seen = feed
        return [self.output]


def test_detector_uses_session_input_size_and_decodes(tmp_path):
    session = _FakeSession(_raw([(160.0, 160.0, 80.0, 80.0, 0.95)]))
    detector = OnnxPersonDetector(tmp_path / "model.onnx", session=session)

    detections = detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))

    assert detector.input_size == 320
    assert session.seen["images"].shape == (1, 3, 320, 320)
    assert len(detections) == 1


def test_detector_requires_a_local_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        OnnxPersonDetector(tmp_path / "absent.onnx")
