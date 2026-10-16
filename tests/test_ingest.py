from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.fixtures import write_image, write_video
from visionops.ingest import VideoDecodeError, ingest
from visionops.jsonl import read_jsonl
from visionops.schemas import FrameRecord


def test_video_ingest_uses_collision_safe_names_and_timestamps(tmp_path: Path) -> None:
    left = tmp_path / "left" / "crowd.mp4"
    right = tmp_path / "right" / "crowd.mp4"
    write_video(left)
    write_video(right)

    manifest = ingest([left, right], tmp_path / "out", stride=2, max_frames=None, seed=7)
    records = read_jsonl(manifest, FrameRecord)

    assert len(records) == 6
    assert len({record.image_id for record in records}) == 6
    assert {record.width for record in records} == {64}
    assert {record.height for record in records} == {48}
    assert [record.timestamp_s for record in records[:3]] == pytest.approx([0.0, 0.2, 0.4])
    assert all((manifest.parent / record.image_path).is_file() for record in records)


def test_image_directory_group_and_max_frames_are_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "images"
    for index in range(6):
        write_image(source / f"frame-{index}.jpg", index * 20)

    first = ingest([source], tmp_path / "first", 1, 3, 23, group="station-a")
    second = ingest([source], tmp_path / "second", 1, 3, 23, group="station-a")
    first_records = read_jsonl(first, FrameRecord)
    second_records = read_jsonl(second, FrameRecord)

    assert len(first_records) == 3
    assert [record.image_id for record in first_records] == [
        record.image_id for record in second_records
    ]
    assert {record.split_group for record in first_records} == {"station-a"}


def test_ingest_rejects_invalid_or_missing_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stride"):
        ingest([tmp_path], tmp_path / "out", 0, None, 0)
    with pytest.raises(FileNotFoundError, match="missing.mp4"):
        ingest([tmp_path / "missing.mp4"], tmp_path / "out", 1, None, 0)


def test_ingest_reports_undecodable_image_path(tmp_path: Path) -> None:
    source = tmp_path / "images"
    source.mkdir()
    bad = source / "bad.jpg"
    bad.write_bytes(b"not-an-image")

    with pytest.raises(ValueError, match="bad.jpg"):
        ingest([source], tmp_path / "out", 1, None, 0)


def test_ingest_reports_truncated_video_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "truncated.mp4"
    video.write_bytes(b"present")

    class TruncatedCapture:
        def __init__(self, _: str) -> None:
            self.reads = 0

        def isOpened(self) -> bool:
            return True

        def get(self, key: int) -> float:
            values = {
                cv2.CAP_PROP_FPS: 10.0,
                cv2.CAP_PROP_FRAME_WIDTH: 64.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 48.0,
                cv2.CAP_PROP_FRAME_COUNT: 3.0,
            }
            return values[key]

        def read(self) -> tuple[bool, np.ndarray | None]:
            self.reads += 1
            if self.reads == 1:
                return True, np.zeros((48, 64, 3), dtype=np.uint8)
            return False, None

        def release(self) -> None:
            return None

    monkeypatch.setattr(cv2, "VideoCapture", TruncatedCapture)

    with pytest.raises(VideoDecodeError, match="truncated.mp4"):
        ingest([video], tmp_path / "out", 1, None, 0)
