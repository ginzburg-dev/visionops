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
