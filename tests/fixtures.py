import hashlib
from pathlib import Path

import cv2
import numpy as np


def write_video(path: Path, *, frame_count: int = 5, fps: float = 10.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.full((48, 64, 3), index * 30, dtype=np.uint8)
        cv2.putText(frame, str(index), (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        writer.write(frame)
    writer.release()


def write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((24, 32, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def make_frame(image_id: str, *, group: str = "group-a", path: str | None = None):
    from visionops.schemas import FrameRecord

    index = int(image_id.rsplit("-", 1)[-1])
    return FrameRecord(
        image_id=image_id,
        image_path=path or f"images/{image_id}.jpg",
        source_path=f"/data/{group}.mp4",
        source_id=group,
        frame_index=index,
        timestamp_s=0.0,
        width=32,
        height=24,
        split_group=group,
        content_sha256=hashlib.sha256(image_id.encode("utf-8")).hexdigest(),
    )


def make_prediction(image_id: str, confidences=(0.9,), **kwargs):
    from visionops.schemas import Detection, PredictionRecord

    detections = tuple(
        Detection(
            class_id=0,
            class_name="person",
            confidence=value,
            bbox_xyxy_px=(1.0 + index, 2.0, 11.0 + index, 20.0),
        )
        for index, value in enumerate(confidences)
    )
    return PredictionRecord(
        image=make_frame(image_id, **kwargs),
        model_id="fake-model",
        model_sha256="b" * 64,
        confidence_threshold=0.25,
        iou_threshold=0.7,
        detections=detections,
        person_count=len(detections),
    )


def write_fixture_onnx(path: Path, predictions, input_size: int = 640) -> Path:
    """Write a tiny constant-output ONNX model shaped like YOLO output for runtime tests."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    array = np.asarray(predictions, dtype=np.float32).reshape(1, *np.shape(predictions))
    tensor = numpy_helper.from_array(array, name="const_output")
    node = helper.make_node("Constant", inputs=[], outputs=["output0"], value=tensor)
    graph = helper.make_graph(
        [node],
        "visionops_fixture",
        [
            helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, [1, 3, input_size, input_size]
            )
        ],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(array.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model.SerializeToString())
    return path
