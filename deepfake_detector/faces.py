"""Face detection backends and a simple single-subject tracker."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from deepfake_detector.config import FaceBackend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Box:
    """Face box in pixel coordinates of the original frame."""

    top: int
    right: int
    bottom: int
    left: int

    @property
    def area(self) -> int:
        return max(0, self.bottom - self.top) * max(0, self.right - self.left)

    def iou(self, other: Box) -> float:
        top, left = max(self.top, other.top), max(self.left, other.left)
        bottom, right = min(self.bottom, other.bottom), min(self.right, other.right)
        inter = max(0, bottom - top) * max(0, right - left)
        union = self.area + other.area - inter
        return inter / union if union else 0.0

    def clip(self, height: int, width: int) -> Box:
        return Box(
            top=max(0, self.top),
            right=min(width, self.right),
            bottom=min(height, self.bottom),
            left=max(0, self.left),
        )

    def as_list(self) -> list[int]:
        return [self.top, self.right, self.bottom, self.left]


class FaceDetector(Protocol):
    name: str

    def detect(self, frame_bgr: np.ndarray) -> list[Box]: ...


class _Downscaling:
    """Run detection on a downscaled copy of the frame and map boxes back."""

    def __init__(self, max_side: int):
        self.max_side = max_side

    def _scaled(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        scale = min(1.0, self.max_side / max(h, w)) if self.max_side > 0 else 1.0
        if scale < 1.0:
            frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        return frame, scale

    @staticmethod
    def _unscale(box: tuple[int, int, int, int], scale: float) -> Box:
        top, right, bottom, left = (round(v / scale) for v in box)
        return Box(top, right, bottom, left)


class FaceRecognitionDetector(_Downscaling):
    """dlib HOG detector via ``face_recognition``; the backend the checkpoint was trained with."""

    name = "face_recognition"

    def __init__(self, max_side: int = 640):
        super().__init__(max_side)
        import face_recognition  # noqa: F401  (optional dependency, fail early)

        self._fr = face_recognition

    def detect(self, frame_bgr: np.ndarray) -> list[Box]:
        small, scale = self._scaled(frame_bgr)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        return [self._unscale(loc, scale) for loc in self._fr.face_locations(rgb)]


class OpenCVHaarDetector(_Downscaling):
    """Haar cascade bundled with OpenCV. No extra install, lower recall than dlib."""

    name = "opencv"

    def __init__(self, max_side: int = 640):
        super().__init__(max_side)
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError(f"failed to load Haar cascade from {path}")

    def detect(self, frame_bgr: np.ndarray) -> list[Box]:
        small, scale = self._scaled(frame_bgr)
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        found = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        return [self._unscale((y, x + w, y + h, x), scale) for (x, y, w, h) in found]


def build_face_detector(backend: FaceBackend = "auto", max_side: int = 640) -> FaceDetector:
    if backend in ("auto", "face_recognition"):
        try:
            return FaceRecognitionDetector(max_side)
        except ImportError:
            if backend == "face_recognition":
                raise
            logger.warning(
                "face_recognition is not installed; falling back to the OpenCV Haar cascade. "
                "The checkpoint was trained on dlib crops, so accuracy may drop."
            )
    if backend in ("auto", "opencv"):
        return OpenCVHaarDetector(max_side)
    raise ValueError(f"unknown face backend: {backend!r}")


class FaceTracker:
    """Follow one subject across frames.

    The original code took ``faces[0]``, whose order is arbitrary, so in multi-person
    videos the crop could jump between people. We start from the largest face and then
    prefer the box overlapping the previous one most.
    """

    def __init__(self, detector: FaceDetector, detect_every: int = 1, min_iou: float = 0.1):
        self.detector = detector
        self.detect_every = max(1, detect_every)
        self.min_iou = min_iou
        self._last: Box | None = None
        self._since_detect = 0

    def reset(self) -> None:
        self._last = None
        self._since_detect = 0

    def update(self, frame_bgr: np.ndarray) -> Box | None:
        if self._last is not None and self._since_detect < self.detect_every - 1:
            self._since_detect += 1
            return self._last

        self._since_detect = 0
        boxes = [b for b in self.detector.detect(frame_bgr) if b.area > 0]
        if not boxes:
            self._last = None
            return None

        h, w = frame_bgr.shape[:2]
        if self._last is not None:
            best = max(boxes, key=self._last.iou)
            if self._last.iou(best) >= self.min_iou:
                self._last = best.clip(h, w)
                return self._last
        self._last = max(boxes, key=lambda b: b.area).clip(h, w)
        return self._last
