from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

SCHEMA_VERSION = 1


def _require_schema(data: dict[str, Any]) -> None:
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {data.get('schema_version')!r}")


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sha256(value: str, name: str) -> str:
    result = str(value).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    return result


@dataclass(frozen=True, slots=True)
class FrameRecord:
    image_id: str
    image_path: str
    source_path: str
    source_id: str
    frame_index: int
    timestamp_s: float
    width: int
    height: int
    split_group: str
    content_sha256: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        if not self.image_id or not self.image_path or not self.source_id or not self.split_group:
            raise ValueError("frame identifiers and paths must not be empty")
        if self.frame_index < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("frame index and dimensions must be positive")
        object.__setattr__(self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s"))
        if self.timestamp_s < 0:
            raise ValueError("timestamp_s must be non-negative")
        object.__setattr__(self, "content_sha256", _sha256(self.content_sha256, "content_sha256"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "image_id": self.image_id,
            "image_path": self.image_path,
            "source_path": self.source_path,
            "source_id": self.source_id,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "width": self.width,
            "height": self.height,
            "split_group": self.split_group,
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameRecord:
        _require_schema(data)
        return cls(**data)


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy_px: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        confidence = _finite(self.confidence, "confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        bbox = tuple(_finite(value, "bbox_xyxy_px") for value in self.bbox_xyxy_px)
        if len(bbox) != 4:
            raise ValueError("bbox_xyxy_px must contain four values")
        if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
            raise ValueError("bbox_xyxy_px must use x1,y1,x2,y2 ordering")
        if self.class_id < 0 or not self.class_name:
            raise ValueError("detection class must be valid")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "bbox_xyxy_px", bbox)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox_xyxy_px": list(self.bbox_xyxy_px),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Detection:
        payload = dict(data)
        payload["bbox_xyxy_px"] = tuple(payload["bbox_xyxy_px"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    image: FrameRecord
    model_id: str
    model_sha256: str
    confidence_threshold: float
    iou_threshold: float
    detections: tuple[Detection, ...]
    person_count: int
    warnings: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        confidence = _finite(self.confidence_threshold, "confidence_threshold")
        iou = _finite(self.iou_threshold, "iou_threshold")
        if not 0 <= confidence <= 1 or not 0 <= iou <= 1:
            raise ValueError("thresholds must be between 0 and 1")
        actual = sum(item.class_name == "person" for item in self.detections)
        if self.person_count != actual:
            raise ValueError("person_count must equal retained person detections")
        object.__setattr__(self, "model_sha256", _sha256(self.model_sha256, "model_sha256"))
        object.__setattr__(self, "confidence_threshold", confidence)
        object.__setattr__(self, "iou_threshold", iou)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "image": self.image.to_dict(),
            "model_id": self.model_id,
            "model_sha256": self.model_sha256,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "detections": [item.to_dict() for item in self.detections],
            "person_count": self.person_count,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PredictionRecord:
        _require_schema(data)
        payload = dict(data)
        payload["image"] = FrameRecord.from_dict(payload["image"])
        payload["detections"] = tuple(Detection.from_dict(item) for item in payload["detections"])
        payload["warnings"] = tuple(payload.get("warnings", ()))
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class SelectionRecord:
    prediction: PredictionRecord
    rank: int
    strategy: str
    uncertainty_score: float
    diversity_score: float
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        if self.rank < 1 or not self.strategy:
            raise ValueError("selection rank and strategy must be valid")
        for name in ("uncertainty_score", "diversity_score"):
            value = _finite(getattr(self, name), name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "prediction": self.prediction.to_dict(),
            "rank": self.rank,
            "strategy": self.strategy,
            "uncertainty_score": self.uncertainty_score,
            "diversity_score": self.diversity_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SelectionRecord:
        _require_schema(data)
        payload = dict(data)
        payload["prediction"] = PredictionRecord.from_dict(payload["prediction"])
        return cls(**payload)
