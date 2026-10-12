"""Video decoding helpers built on OpenCV."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class VideoReadError(RuntimeError):
    """The file could not be opened or decoded as a video."""


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def probe(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoReadError(f"cannot open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if frame_count <= 0:
            # Some containers do not report a frame count; count by decoding.
            frame_count = sum(1 for _ in _read(cap))
        if frame_count <= 0:
            raise VideoReadError(f"video has no decodable frames: {path}")
        return VideoInfo(str(path), fps if fps > 0 else 25.0, frame_count, width, height)
    finally:
        cap.release()


def iter_frames(path: str | Path, start: int = 0, stop: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, bgr_frame)`` for frames in ``[start, stop)``."""

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoReadError(f"cannot open video: {path}")
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        index = start
        for frame in _read(cap):
            if stop is not None and index >= stop:
                break
            yield index, frame
            index += 1
    finally:
        cap.release()


def plan_clips(frame_count: int, num_clips: int, min_span: int) -> list[tuple[int, int]]:
    """Partition ``[0, frame_count)`` into up to ``num_clips`` equal, contiguous segments.

    A clip is read from the start of its segment until enough face crops are collected
    or the segment ends, so segments shorter than ``min_span`` frames are not created.
    """

    if frame_count <= 0:
        return []
    n = max(1, min(num_clips, frame_count // max(1, min_span)))
    edges = np.linspace(0, frame_count, n + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n)]


def _read(cap: cv2.VideoCapture) -> Iterator[np.ndarray]:
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield frame
