from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from visionops.hashing import file_sha256
from visionops.schemas import Detection

PERSON_CLASS_NAME = "person"
DEFAULT_INPUT_SIZE = 640
PAD_VALUE = 114


class OnnxOutputLayoutError(ValueError):
    """Raised when an ONNX detection output cannot be mapped onto the YOLO layout."""


@dataclass(frozen=True, slots=True)
class PreprocessedImage:
    tensor: np.ndarray
    scale: float
    pad_x: float
    pad_y: float
    original_width: int
    original_height: int
    input_size: int


def letterbox(image_bgr: np.ndarray, size: int = DEFAULT_INPUT_SIZE) -> PreprocessedImage:
    """Resize with preserved aspect ratio, pad to a square, and convert BGR to normalized RGB."""
    if image_bgr is None or image_bgr.size == 0 or image_bgr.ndim != 3:
        raise ValueError("letterbox expects a non-empty HxWx3 BGR image")
    if size <= 0:
        raise ValueError("letterbox size must be greater than zero")
    height, width = image_bgr.shape[:2]
    scale = min(size / width, size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image_bgr, (new_width, new_height), interpolation=interpolation)
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    pad_x = (size - new_width) / 2.0
    pad_y = (size - new_height) / 2.0
    top = int(round(pad_y - 0.1))
    left = int(round(pad_x - 0.1))
    canvas[top : top + new_height, left : left + new_width] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...])
    return PreprocessedImage(
        tensor=tensor,
        scale=scale,
        pad_x=float(left),
        pad_y=float(top),
        original_width=width,
        original_height=height,
        input_size=size,
    )


def _as_predictions(output: np.ndarray) -> np.ndarray:
    """Return predictions shaped [N, attributes] from either YOLO output layout."""
    array = np.asarray(output, dtype=np.float32)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise OnnxOutputLayoutError(f"unsupported ONNX output shape: {array.shape}")
        array = array[0]
    if array.ndim != 2:
        raise OnnxOutputLayoutError(f"unsupported ONNX output shape: {np.shape(output)}")
    rows, columns = array.shape
    rows_are_attributes = rows >= 5
    columns_are_attributes = columns >= 5
    if columns_are_attributes and (not rows_are_attributes or columns <= rows):
        return array
    if rows_are_attributes:
        return array.transpose()
    raise OnnxOutputLayoutError(f"unsupported ONNX output shape: {np.shape(output)}")


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    order = np.argsort(-scores, kind="stable")
    areas = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        overlap = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        union = areas[current] + areas[rest] - overlap
        iou = np.where(union > 0, overlap / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_threshold]
    return keep


def decode_yolo_output(
    output: np.ndarray,
    metadata: PreprocessedImage,
    confidence: float,
    iou: float,
    person_class_id: int = 0,
) -> list[Detection]:
    """Decode raw YOLO output into person detections in original image pixels."""
    predictions = _as_predictions(output)
    attributes = predictions.shape[1]
    if attributes < 5 or person_class_id >= attributes - 4:
        raise OnnxOutputLayoutError(
            f"output with {attributes} attributes has no class {person_class_id}"
        )
    scores = predictions[:, 4 + person_class_id]
    finite = np.all(np.isfinite(predictions[:, :4]), axis=1) & np.isfinite(scores)
    retained = finite & (scores >= confidence)
    if not np.any(retained):
        return []
    centers = predictions[retained, :4]
    scores = scores[retained]
    boxes = np.empty_like(centers)
    boxes[:, 0] = centers[:, 0] - centers[:, 2] / 2.0
    boxes[:, 1] = centers[:, 1] - centers[:, 3] / 2.0
    boxes[:, 2] = centers[:, 0] + centers[:, 2] / 2.0
    boxes[:, 3] = centers[:, 1] + centers[:, 3] / 2.0

    keep = non_max_suppression(boxes, scores, iou)
    boxes = boxes[keep]
    scores = scores[keep]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - metadata.pad_x) / metadata.scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - metadata.pad_y) / metadata.scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, metadata.original_width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, metadata.original_height)

    detections: list[Detection] = []
    for box, score in zip(boxes, scores, strict=True):
        if box[2] - box[0] <= 0 or box[3] - box[1] <= 0:
            continue
        detections.append(
            Detection(
                class_id=int(person_class_id),
                class_name=PERSON_CLASS_NAME,
                confidence=float(min(max(score, 0.0), 1.0)),
                bbox_xyxy_px=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            )
        )
    return detections


class OnnxPersonDetector:
    """ONNX Runtime person detector. Production inference never falls back to PyTorch."""

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.25,
        iou: float = 0.7,
        providers: list[str] | None = None,
        session: Any | None = None,
        person_class_id: int = 0,
    ) -> None:
        path = Path(model_path)
        if session is None and not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        self.model_path = path
        self.model_id = path.name
        self.model_sha256 = file_sha256(path) if path.is_file() else "0" * 64
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.person_class_id = int(person_class_id)
        self.providers = providers or ["CPUExecutionProvider"]
        self._session = session or self._create_session()
        self._input_name = self._session.get_inputs()[0].name
        self.input_size = self._resolve_input_size()

    def _create_session(self) -> Any:
        import onnxruntime  # imported lazily so the CLI stays importable without a model

        return onnxruntime.InferenceSession(str(self.model_path), providers=self.providers)

    def _resolve_input_size(self) -> int:
        shape = list(getattr(self._session.get_inputs()[0], "shape", []) or [])
        candidate = shape[-1] if shape else None
        return (
            int(candidate) if isinstance(candidate, int) and candidate > 0 else (DEFAULT_INPUT_SIZE)
        )

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        metadata = letterbox(image_bgr, self.input_size)
        outputs = self._session.run(None, {self._input_name: metadata.tensor})
        if not outputs:
            raise OnnxOutputLayoutError("ONNX session returned no outputs")
        return decode_yolo_output(
            np.asarray(outputs[0]),
            metadata,
            self.confidence,
            self.iou,
            self.person_class_id,
        )
