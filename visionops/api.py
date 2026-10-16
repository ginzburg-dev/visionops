from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse

from visionops.inference import OnnxPersonDetector

MAX_UPLOAD_BYTES = 10_000_000
ALLOWED_CONTENT_TYPES = ("image/jpeg", "image/png", "image/webp", "image/bmp")


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content={"error": {"code": code, "message": message}}
    )


def create_app(
    model_path: str | Path,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
    detector_factory: Callable[[Path], Any] | None = None,
) -> FastAPI:
    """Build the ONNX-only serving app. There is no PyTorch fallback at serving time."""
    path = Path(model_path)
    factory = detector_factory or OnnxPersonDetector

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            app.state.detector = factory(path)
            app.state.load_error = None
        except Exception as error:  # surfaced as 503 instead of crashing the worker
            app.state.detector = None
            app.state.load_error = f"{type(error).__name__}: {error}"
        yield
        app.state.detector = None

    app = FastAPI(title="VisionOps person detection", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(ValueError)
    async def _value_error(request: Request, error: ValueError) -> JSONResponse:
        return _error(400, "invalid_request", str(error))

    @app.get("/health")
    async def health() -> JSONResponse:
        detector = app.state.detector
        if detector is None:
            return _error(503, "model_unavailable", app.state.load_error or "model not loaded")
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "model_id": detector.model_id,
                "model_sha256": detector.model_sha256,
                "input_size": detector.input_size,
                "confidence_threshold": detector.confidence,
                "iou_threshold": detector.iou,
                "classes": ["person"],
                "max_upload_bytes": int(max_upload_bytes),
            },
        )

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> JSONResponse:
        detector = app.state.detector
        if detector is None:
            return _error(503, "model_unavailable", app.state.load_error or "model not loaded")
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            return _error(
                415,
                "unsupported_media_type",
                f"content type {file.content_type} is not supported; "
                f"use one of {', '.join(ALLOWED_CONTENT_TYPES)}",
            )
        payload = await file.read(int(max_upload_bytes) + 1)
        if len(payload) > max_upload_bytes:
            return _error(413, "upload_too_large", f"upload exceeds {int(max_upload_bytes)} bytes")
        if not payload:
            return _error(400, "empty_upload", "the uploaded file is empty")
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return _error(400, "undecodable_image", "the uploaded bytes are not a decodable image")
        try:
            detections = detector.detect(image)
        except ValueError as error:
            return _error(400, "inference_failed", str(error))
        height, width = image.shape[:2]
        return JSONResponse(
            status_code=200,
            content={
                "person_count": len(detections),
                "detections": [item.to_dict() for item in detections],
                "image": {"width": int(width), "height": int(height)},
                "model_id": detector.model_id,
                "model_sha256": detector.model_sha256,
                "confidence_threshold": detector.confidence,
                "iou_threshold": detector.iou,
            },
        )

    return app
