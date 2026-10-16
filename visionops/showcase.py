from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from visionops.hashing import file_sha256
from visionops.schemas import Detection
from visionops.training import environment_versions, write_json

MIN_DURATION_S = 8.0
MAX_DURATION_S = 15.0
MIN_WIDTH = 1920
MIN_HEIGHT = 1080
BOX_COLOR = (0, 220, 255)
TEXT_COLOR = (16, 16, 16)
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(frozen=True, slots=True)
class ShowcaseArtifacts:
    video_path: Path
    image_path: Path
    metadata_path: Path
    frame_count: int
    fps: float
    duration_s: float
    max_person_count: int


def clip_box(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    """Clip a detection box so every drawn pixel stays inside the frame."""
    x1 = int(round(min(max(bbox[0], 0.0), width - 1)))
    y1 = int(round(min(max(bbox[1], 0.0), height - 1)))
    x2 = int(round(min(max(bbox[2], 0.0), width - 1)))
    y2 = int(round(min(max(bbox[3], 0.0), height - 1)))
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def draw_overlay(frame: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    """Draw person boxes and a count banner without writing outside the frame."""
    height, width = frame.shape[:2]
    canvas = frame.copy()
    scale = max(0.4, min(width, height) / 1400.0)
    thickness = max(1, int(round(scale * 2)))
    for detection in detections:
        x1, y1, x2, y2 = clip_box(detection.bbox_xyxy_px, width, height)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), BOX_COLOR, thickness)
        label = f"person {detection.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label, FONT, scale, thickness)
        label_top = max(0, y1 - text_height - baseline)
        label_right = min(width - 1, x1 + text_width + 4)
        cv2.rectangle(canvas, (x1, label_top), (label_right, y1), BOX_COLOR, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 2, max(text_height, y1 - baseline)),
            FONT,
            scale,
            TEXT_COLOR,
            thickness,
            cv2.LINE_AA,
        )
    banner = f"person_count: {len(detections)}"
    (banner_width, banner_height), baseline = cv2.getTextSize(banner, FONT, scale, thickness)
    cv2.rectangle(
        canvas,
        (0, 0),
        (min(width - 1, banner_width + 16), min(height - 1, banner_height + baseline + 12)),
        BOX_COLOR,
        -1,
    )
    cv2.putText(
        canvas,
        banner,
        (8, banner_height + 6),
        FONT,
        scale,
        TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )
    return canvas


def make_showcase(
    video_path: str | Path,
    detector: Any,
    out_dir: str | Path,
    start_s: float = 0.0,
    duration_s: float = 10.0,
    *,
    source_url: str | None = None,
    license_name: str | None = None,
    attribution: str | None = None,
    name: str = "crowd-showcase",
) -> ShowcaseArtifacts:
    """Render an annotated clip from local footage. Nothing is downloaded here."""
    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(f"showcase source video not found: {source}")
    if not MIN_DURATION_S <= duration_s <= MAX_DURATION_S:
        raise ValueError(
            f"showcase duration must be between {MIN_DURATION_S:.0f} and "
            f"{MAX_DURATION_S:.0f} seconds, got {duration_s}"
        )
    if start_s < 0:
        raise ValueError("showcase start must not be negative")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"could not open showcase source: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError(f"showcase source has invalid frame rate: {source}")
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        capture.release()
        raise ValueError(
            f"showcase source must be at least {MIN_WIDTH}x{MIN_HEIGHT}, "
            f"got {width}x{height}: {source}"
        )

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    video_out = destination / f"{name}.mp4"
    image_out = destination / f"{name}.png"
    metadata_out = destination / f"{name}.json"

    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"could not open showcase writer: {video_out}")

    requested = int(round(duration_s * fps))
    start_frame = int(round(start_s * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    person_counts: list[int] = []
    best_frame: np.ndarray | None = None
    best_count = -1
    try:
        while len(person_counts) < requested:
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                break
            detections = list(detector.detect(frame))
            annotated = draw_overlay(frame, detections)
            writer.write(annotated)
            person_counts.append(len(detections))
            if len(detections) > best_count:
                best_count = len(detections)
                best_frame = annotated
    finally:
        writer.release()
        capture.release()

    if not person_counts or best_frame is None:
        raise ValueError(f"no frames were decoded from {source} at start {start_s}s")
    if len(person_counts) < requested:
        raise ValueError(
            f"showcase source ran out after {len(person_counts)} of {requested} frames: {source}"
        )
    if not cv2.imwrite(str(image_out), best_frame):
        raise RuntimeError(f"could not write showcase still: {image_out}")

    frame_count = len(person_counts)
    write_json(
        metadata_out,
        {
            "video_path": str(video_out),
            "image_path": str(image_out),
            "source_path": str(source),
            "source_sha256": file_sha256(source),
            "source_url": source_url,
            "license": license_name,
            "attribution": attribution,
            "start_s": float(start_s),
            "duration_s": frame_count / fps,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "person_count_per_frame": person_counts,
            "max_person_count": max(person_counts),
            "mean_person_count": sum(person_counts) / frame_count,
            "model_id": getattr(detector, "model_id", None),
            "model_sha256": getattr(detector, "model_sha256", None),
            "confidence_threshold": getattr(detector, "confidence", None),
            "iou_threshold": getattr(detector, "iou", None),
            "classes": ["person"],
            "versions": environment_versions(),
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    return ShowcaseArtifacts(
        video_path=video_out,
        image_path=image_out,
        metadata_path=metadata_out,
        frame_count=frame_count,
        fps=fps,
        duration_s=frame_count / fps,
        max_person_count=max(person_counts),
    )
