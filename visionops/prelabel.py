from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from visionops.hashing import file_sha256
from visionops.jsonl import read_jsonl, write_jsonl
from visionops.schemas import Detection, FrameRecord, PredictionRecord

PERSON_CLASS_NAME = "person"


class PersonDetector(Protocol):
    confidence: float
    iou: float

    def detect(self, image_bgr: np.ndarray) -> Sequence[Detection]: ...


def _as_array(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=np.float64)
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _class_name(names: Any, index: int) -> str:
    try:
        return str(names[index])
    except (KeyError, IndexError, TypeError):
        return ""


def normalize_boxes(
    boxes_xyxy: Any,
    confidences: Any,
    class_ids: Any,
    names: Any,
    *,
    confidence: float,
    width: int,
    height: int,
) -> list[Detection]:
    """Normalize raw detector output into clipped, person-only detections."""
    raw_boxes = _as_array(boxes_xyxy)
    boxes = raw_boxes.reshape(-1, 4) if raw_boxes.size else np.zeros((0, 4))
    scores = _as_array(confidences).reshape(-1)
    classes = _as_array(class_ids).reshape(-1)
    if not (len(boxes) == len(scores) == len(classes)):
        raise ValueError(
            f"detector returned mismatched arrays: {len(boxes)} boxes, "
            f"{len(scores)} scores, {len(classes)} classes"
        )
    detections: list[Detection] = []
    for box, score, class_id in zip(boxes, scores, classes, strict=True):
        if not np.all(np.isfinite(box)) or not np.isfinite(score):
            continue
        index = int(class_id)
        if _class_name(names, index) != PERSON_CLASS_NAME or float(score) < confidence:
            continue
        x1 = float(np.clip(min(box[0], box[2]), 0.0, width))
        y1 = float(np.clip(min(box[1], box[3]), 0.0, height))
        x2 = float(np.clip(max(box[0], box[2]), 0.0, width))
        y2 = float(np.clip(max(box[1], box[3]), 0.0, height))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        detections.append(
            Detection(
                class_id=index,
                class_name=PERSON_CLASS_NAME,
                confidence=min(max(float(score), 0.0), 1.0),
                bbox_xyxy_px=(x1, y1, x2, y2),
            )
        )
    return detections


class UltralyticsPersonDetector:
    """Local-weight Ultralytics adapter; it never resolves model names or remote URLs."""

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.25,
        iou: float = 0.7,
        device: str = "cpu",
        predictor: Callable[[np.ndarray], Any] | None = None,
    ) -> None:
        path = Path(model_path)
        if predictor is None and not path.is_file():
            raise FileNotFoundError(f"local model weight not found: {path}")
        self.model_path = path
        self.model_id = path.name
        self.model_sha256 = file_sha256(path) if path.is_file() else "0" * 64
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.device = device
        self._predictor = predictor
        self._model: Any = None

    def _predict(self, image_bgr: np.ndarray) -> Any:
        if self._predictor is not None:
            return self._predictor(image_bgr)
        if self._model is None:
            from ultralytics import YOLO  # lazy: local training dependency only

            self._model = YOLO(str(self.model_path))
        return self._model.predict(
            source=image_bgr,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        result = self._predict(image_bgr)
        if isinstance(result, (list, tuple)):
            if not result:
                return []
            result = result[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        height, width = image_bgr.shape[:2]
        return normalize_boxes(
            getattr(boxes, "xyxy", None),
            getattr(boxes, "conf", None),
            getattr(boxes, "cls", None),
            getattr(result, "names", {0: PERSON_CLASS_NAME}),
            confidence=self.confidence,
            width=width,
            height=height,
        )


def prelabel_manifest(
    manifest_path: str | Path,
    detector: PersonDetector,
    out_path: str | Path,
    model_id: str,
    model_sha256: str,
) -> Path:
    manifest = Path(manifest_path)
    frames = read_jsonl(manifest, FrameRecord)
    records: list[PredictionRecord] = []
    for frame in frames:
        image_path = manifest.parent / frame.image_path
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"could not decode manifest image: {image_path}")
        detections = tuple(detector.detect(image))
        records.append(
            PredictionRecord(
                image=frame,
                model_id=model_id,
                model_sha256=model_sha256,
                confidence_threshold=detector.confidence,
                iou_threshold=detector.iou,
                detections=detections,
                person_count=len(detections),
                warnings=() if detections else ("no person detections retained",),
            )
        )
    return write_jsonl(out_path, records)
