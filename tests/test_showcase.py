import json

import cv2
import numpy as np
import pytest

from visionops.hashing import file_sha256
from visionops.schemas import Detection
from visionops.showcase import clip_box, draw_overlay, make_showcase

FPS = 5.0


class _Detector:
    model_id = "person.onnx"
    model_sha256 = "a" * 64
    confidence = 0.25
    iou = 0.7

    def __init__(self, count=2):
        self.count = count

    def detect(self, frame):
        return [
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.8,
                bbox_xyxy_px=(10.0 + 40 * index, 20.0, 90.0 + 40 * index, 300.0),
            )
            for index in range(self.count)
        ]


def _video(path, *, width=1920, height=1080, seconds=9.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
    assert writer.isOpened()
    frame = np.full((height, width, 3), 60, dtype=np.uint8)
    for index in range(int(round(seconds * FPS))):
        frame[:] = 40 + (index % 5) * 10
        writer.write(frame)
    writer.release()
    return path


def test_duration_outside_the_window_is_rejected(tmp_path):
    source = _video(tmp_path / "src.mp4")
    for duration in (4.0, 20.0):
        with pytest.raises(ValueError, match="duration must be between 8 and 15"):
            make_showcase(source, _Detector(), tmp_path / "out", 0.0, duration)


def test_source_below_1080p_is_rejected(tmp_path):
    source = _video(tmp_path / "small.mp4", width=640, height=360, seconds=9.0)
    with pytest.raises(ValueError, match="at least 1920x1080"):
        make_showcase(source, _Detector(), tmp_path / "out", 0.0, 8.0)


def test_clip_box_stays_inside_the_frame():
    assert clip_box((-20.0, -5.0, 5000.0, 4000.0), 1920, 1080) == (0, 0, 1919, 1079)


def test_overlay_keeps_frame_shape_and_marks_detections():
    frame = np.full((1080, 1920, 3), 30, dtype=np.uint8)
    annotated = draw_overlay(frame, _Detector(1).detect(frame))
    assert annotated.shape == frame.shape
    assert annotated.dtype == frame.dtype
    assert not np.array_equal(annotated, frame)
    assert frame[0, 0].tolist() == [30, 30, 30]


def test_showcase_renders_synchronized_clip_with_provenance(tmp_path):
    source = _video(tmp_path / "src.mp4")
    artifacts = make_showcase(
        source,
        _Detector(3),
        tmp_path / "out",
        start_s=0.0,
        duration_s=8.0,
        source_url="https://example.invalid/clip",
        license_name="Example licence",
        attribution="Example author",
    )

    assert artifacts.frame_count == 40
    assert artifacts.fps == FPS
    assert artifacts.duration_s == pytest.approx(8.0)
    assert artifacts.max_person_count == 3
    assert artifacts.video_path.is_file()
    assert artifacts.image_path.is_file()

    capture = cv2.VideoCapture(str(artifacts.video_path))
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(FPS)
    assert int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))) == 40
    assert int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))) == 1920
    capture.release()

    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["source_sha256"] == file_sha256(source)
    assert metadata["frame_count"] == 40
    assert metadata["fps"] == FPS
    assert metadata["max_person_count"] == 3
    assert metadata["person_count_per_frame"][:3] == [3, 3, 3]
    assert metadata["model_sha256"] == "a" * 64
    assert metadata["source_url"] == "https://example.invalid/clip"
    assert metadata["license"] == "Example licence"
    assert metadata["attribution"] == "Example author"
    assert metadata["versions"]["python"]


def test_short_source_reports_missing_frames(tmp_path):
    source = _video(tmp_path / "short.mp4", seconds=8.0)
    with pytest.raises(ValueError, match="ran out after"):
        make_showcase(source, _Detector(), tmp_path / "out", start_s=4.0, duration_s=8.0)


def test_missing_source_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="showcase source video not found"):
        make_showcase(tmp_path / "absent.mp4", _Detector(), tmp_path / "out", 0.0, 8.0)
