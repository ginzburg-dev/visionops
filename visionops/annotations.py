from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from visionops.jsonl import read_jsonl
from visionops.schemas import SelectionRecord

FROM_NAME = "label"
TO_NAME = "image"
LABEL = "person"


def _resolve_image(image_path: str, root: Path | None) -> Path:
    candidate = Path(image_path)
    if not candidate.is_absolute() and root is not None:
        candidate = root / candidate
    return candidate


def _rectangle(detection: Any, width: int, height: int, index: int) -> dict[str, Any]:
    x1, y1, x2, y2 = detection.bbox_xyxy_px
    return {
        "id": f"person-{index}",
        "type": "rectanglelabels",
        "from_name": FROM_NAME,
        "to_name": TO_NAME,
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "score": detection.confidence,
        "value": {
            "x": 100.0 * x1 / width,
            "y": 100.0 * y1 / height,
            "width": 100.0 * (x2 - x1) / width,
            "height": 100.0 * (y2 - y1) / height,
            "rotation": 0,
            "rectanglelabels": [LABEL],
        },
    }


def selection_to_tasks(
    records: list[SelectionRecord], root: Path | None = None
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for record in records:
        prediction = record.prediction
        frame = prediction.image
        image = _resolve_image(frame.image_path, root)
        results = [
            _rectangle(detection, frame.width, frame.height, index)
            for index, detection in enumerate(prediction.detections)
        ]
        tasks.append(
            {
                "data": {
                    "image": image.resolve().as_posix(),
                    "image_id": frame.image_id,
                    "split_group": frame.split_group,
                },
                "predictions": [
                    {
                        "model_version": prediction.model_id,
                        "result": results,
                    }
                ],
                "meta": {
                    "rank": record.rank,
                    "strategy": record.strategy,
                    "uncertainty_score": record.uncertainty_score,
                    "diversity_score": record.diversity_score,
                    "model_id": prediction.model_id,
                    "model_sha256": prediction.model_sha256,
                    "confidence_threshold": prediction.confidence_threshold,
                    "iou_threshold": prediction.iou_threshold,
                    "person_count": prediction.person_count,
                    "content_sha256": frame.content_sha256,
                    "review_required": True,
                },
            }
        )
    return tasks


def export_labelstudio(
    selection_path: str | Path, out_path: str | Path, root: str | Path | None = None
) -> Path:
    source = Path(selection_path)
    base = Path(root) if root is not None else source.parent
    records = read_jsonl(source, SelectionRecord)
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = selection_to_tasks(records, base)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return destination
