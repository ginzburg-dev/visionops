from __future__ import annotations

import hashlib
import heapq
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from visionops.jsonl import write_jsonl
from visionops.schemas import FrameRecord

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


class VideoDecodeError(RuntimeError):
    """Raised when decoding stops before the container's declared frame count."""


@dataclass(slots=True)
class _Candidate:
    source_path: Path
    source_id: str
    frame_index: int
    timestamp_s: float
    split_group: str
    image_bgr: np.ndarray


def _source_id(path: Path) -> str:
    resolved = str(path.resolve()).replace("\\", "/").lower()
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}-{digest}"


def _iter_video(path: Path, source_id: str, split_group: str) -> Iterator[_Candidate]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    declared = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if not np.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"video has invalid metadata: {path}")
    decoded = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if image is None or image.size == 0:
                raise VideoDecodeError(f"decoder returned an empty frame at {decoded}: {path}")
            yield _Candidate(path, source_id, decoded, decoded / fps, split_group, image)
            decoded += 1
    finally:
        capture.release()
    if declared > 0 and decoded < declared:
        raise VideoDecodeError(f"video ended after {decoded} of {declared} frames: {path}")


def _iter_directory(path: Path, source_id: str, split_group: str) -> Iterator[_Candidate]:
    images = sorted(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"image directory contains no supported images: {path}")
    for index, image_path in enumerate(images):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"could not decode image: {image_path}")
        yield _Candidate(image_path, source_id, index, 0.0, split_group, image)


def _iter_source(path: Path, group: str | None) -> Iterator[_Candidate]:
    if not path.exists():
        raise FileNotFoundError(f"input does not exist: {path}")
    source_id = _source_id(path)
    split_group = group or source_id
    if path.is_dir():
        yield from _iter_directory(path, source_id, split_group)
        return
    if path.suffix.lower() in VIDEO_SUFFIXES:
        yield from _iter_video(path, source_id, split_group)
        return
    if path.suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"could not decode image: {path}")
        yield _Candidate(path, source_id, 0, 0.0, split_group, image)
        return
    raise ValueError(f"unsupported input type: {path}")


def _priority(candidate: _Candidate, seed: int) -> int:
    value = f"{seed}:{candidate.source_id}:{candidate.frame_index}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _write_candidate(candidate: _Candidate, out_dir: Path) -> FrameRecord:
    image_id = f"{candidate.source_id}-{candidate.frame_index:08d}"
    relative_path = Path("images") / f"{image_id}.jpg"
    destination = out_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", candidate.image_bgr)
    if not ok:
        raise RuntimeError(f"could not encode extracted image: {destination}")
    content = encoded.tobytes()
    destination.write_bytes(content)
    height, width = candidate.image_bgr.shape[:2]
    return FrameRecord(
        image_id=image_id,
        image_path=relative_path.as_posix(),
        source_path=str(candidate.source_path.resolve()),
        source_id=candidate.source_id,
        frame_index=candidate.frame_index,
        timestamp_s=candidate.timestamp_s,
        width=width,
        height=height,
        split_group=candidate.split_group,
        content_sha256=hashlib.sha256(content).hexdigest(),
    )


def ingest(
    inputs: Sequence[Path],
    out_dir: Path,
    stride: int,
    max_frames: int | None,
    seed: int,
    group: str | None = None,
) -> Path:
    if stride <= 0:
        raise ValueError("stride must be greater than zero")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be greater than zero")
    if not inputs:
        raise ValueError("at least one input is required")

    selected: list[_Candidate] = []
    ranked: list[tuple[int, int, _Candidate]] = []
    sequence = 0
    for input_path in inputs:
        for candidate in _iter_source(Path(input_path), group):
            if candidate.frame_index % stride:
                continue
            if max_frames is None:
                selected.append(candidate)
            else:
                priority = _priority(candidate, seed)
                entry = (-priority, sequence, candidate)
                sequence += 1
                if len(ranked) < max_frames:
                    heapq.heappush(ranked, entry)
                elif entry > ranked[0]:
                    heapq.heapreplace(ranked, entry)
    if max_frames is not None:
        selected = [entry[2] for entry in ranked]
    selected.sort(key=lambda item: (item.source_id, item.frame_index))

    output = Path(out_dir)
    records = [_write_candidate(candidate, output) for candidate in selected]
    return write_jsonl(output / "manifest.jsonl", records)
