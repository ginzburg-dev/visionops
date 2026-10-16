from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2

from visionops.hashing import file_sha256
from visionops.inference import DEFAULT_INPUT_SIZE, PERSON_CLASS_NAME, OnnxPersonDetector
from visionops.schemas import Detection
from visionops.training import environment_versions, write_json

OPSET = 12
BOX_TOLERANCE_PX = 2.0
CONFIDENCE_TOLERANCE = 0.02


class ParityError(RuntimeError):
    """Raised when ONNX detections diverge from the source model on identical samples."""


def compare_detections(
    reference: Sequence[Detection],
    candidate: Sequence[Detection],
    *,
    box_tolerance_px: float = BOX_TOLERANCE_PX,
    confidence_tolerance: float = CONFIDENCE_TOLERANCE,
    label: str = "sample",
) -> dict[str, float]:
    if len(reference) != len(candidate):
        raise ParityError(
            f"{label}: ONNX returned {len(candidate)} person detections, "
            f"source model returned {len(reference)}"
        )
    max_box = 0.0
    max_confidence = 0.0
    for index, (left, right) in enumerate(zip(reference, candidate, strict=True)):
        box_delta = max(abs(a - b) for a, b in zip(left.bbox_xyxy_px, right.bbox_xyxy_px))
        confidence_delta = abs(left.confidence - right.confidence)
        max_box = max(max_box, box_delta)
        max_confidence = max(max_confidence, confidence_delta)
        if box_delta > box_tolerance_px or confidence_delta > confidence_tolerance:
            raise ParityError(
                f"{label}: detection {index} diverged by {box_delta:.3f} px and "
                f"{confidence_delta:.4f} confidence "
                f"(source {left.bbox_xyxy_px} @ {left.confidence:.4f}, "
                f"onnx {right.bbox_xyxy_px} @ {right.confidence:.4f})"
            )
    return {
        "detections": float(len(reference)),
        "max_box_delta_px": max_box,
        "max_confidence_delta": max_confidence,
    }


def write_model_manifest(
    out_path: str | Path,
    *,
    model_path: Path,
    source_model_path: Path,
    imgsz: int,
    confidence: float,
    iou: float,
    parity: dict[str, Any],
    dataset_config_sha256: str | None = None,
) -> Path:
    return write_json(
        out_path,
        {
            "model_path": str(model_path),
            "model_sha256": file_sha256(model_path) if model_path.is_file() else None,
            "source_model_path": str(source_model_path),
            "source_model_sha256": (
                file_sha256(source_model_path) if source_model_path.is_file() else None
            ),
            "class_map": {"0": PERSON_CLASS_NAME},
            "preprocessing": {
                "letterbox": True,
                "input_size": int(imgsz),
                "pad_value": 114,
                "channel_order": "RGB",
                "scale": "0-1 float32",
                "layout": "NCHW",
            },
            "thresholds": {"confidence": float(confidence), "iou": float(iou)},
            "opset": OPSET,
            "dataset_config_sha256": dataset_config_sha256,
            "versions": environment_versions(),
            "parity": parity,
        },
    )


def _ultralytics_exporter(model_path: Path) -> Callable[..., Any]:
    def run(**kwargs: Any) -> Any:
        from ultralytics import YOLO  # lazy: export-time dependency only

        return YOLO(str(model_path)).export(**kwargs)

    return run


def export_onnx(
    model_path: str | Path,
    out_path: str | Path,
    imgsz: int = DEFAULT_INPUT_SIZE,
    sample_images: Sequence[str | Path] = (),
    exporter: Callable[..., Any] | None = None,
    source_detector: Any | None = None,
    onnx_detector_factory: Callable[[Path], Any] | None = None,
    confidence: float = 0.25,
    iou: float = 0.7,
    dataset_config_sha256: str | None = None,
) -> Path:
    source = Path(model_path)
    if exporter is None and not source.is_file():
        raise FileNotFoundError(f"source model not found: {source}")
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    run = exporter if exporter is not None else _ultralytics_exporter(source)
    produced = Path(
        str(
            run(
                format="onnx",
                imgsz=int(imgsz),
                opset=OPSET,
                dynamic=False,
                simplify=False,
                device="cpu",
            )
        )
    )
    if produced.resolve() != destination.resolve():
        shutil.move(str(produced), str(destination))

    factory = onnx_detector_factory or (
        lambda path: OnnxPersonDetector(path, confidence=confidence, iou=iou)
    )
    detector = factory(destination)
    if int(getattr(detector, "input_size", imgsz)) != int(imgsz):
        raise ValueError(
            f"exported model expects input size {detector.input_size}, requested {imgsz}"
        )

    parity: dict[str, Any] = {"verified": False, "samples": []}
    if sample_images and source_detector is not None:
        samples = []
        for image_path in sample_images:
            path = Path(image_path)
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise ValueError(f"could not decode parity sample: {path}")
            measured = compare_detections(
                source_detector.detect(image), detector.detect(image), label=path.name
            )
            samples.append({"image": str(path), **measured})
        parity = {
            "verified": True,
            "box_tolerance_px": BOX_TOLERANCE_PX,
            "confidence_tolerance": CONFIDENCE_TOLERANCE,
            "samples": samples,
        }

    write_model_manifest(
        destination.with_name("model-manifest.json"),
        model_path=destination,
        source_model_path=source,
        imgsz=imgsz,
        confidence=confidence,
        iou=iou,
        parity=parity,
        dataset_config_sha256=dataset_config_sha256,
    )
    return destination
