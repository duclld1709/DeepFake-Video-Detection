from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from deepfake_detector.config import DetectorConfig
from deepfake_detector.faces import Box
from deepfake_detector.inference import DeepfakeAnalyzer
from deepfake_detector.model import DeepfakeDetector, ModelSpec


def write_video(path: Path, frames: int = 48, size: tuple[int, int] = (160, 120), fps: float = 24.0) -> Path:
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(0)
    for i in range(frames):
        frame = rng.integers(0, 60, (h, w, 3), dtype=np.uint8)
        x = 30 + (i % 20)
        cv2.rectangle(frame, (x, 30), (x + 60, 100), (180, 160, 140), -1)  # a "face"
        writer.write(frame)
    writer.release()
    return path


class StubDetector:
    """Deterministic detector: a fixed box, optionally only on some frames."""

    name = "stub"

    def __init__(self, box: Box | None = Box(30, 100, 100, 30), every: int = 1):
        self.box, self.every, self.calls = box, every, 0

    def detect(self, frame):
        self.calls += 1
        if self.box is None or (self.calls - 1) % self.every:
            return []
        return [self.box]


@pytest.fixture(scope="session")
def legacy_model() -> DeepfakeDetector:
    torch.manual_seed(0)
    return DeepfakeDetector(ModelSpec(temporal_mode="legacy"), pretrained_backbone=False).eval()


@pytest.fixture(scope="session")
def video(tmp_path_factory) -> Path:
    return write_video(tmp_path_factory.mktemp("videos") / "clip.mp4")


@pytest.fixture
def config() -> DetectorConfig:
    return DetectorConfig(device="cpu", sequence_length=8, num_clips=2, gradcam_frames=3)


@pytest.fixture
def analyzer(legacy_model, config) -> DeepfakeAnalyzer:
    return DeepfakeAnalyzer(legacy_model, config, face_detector=StubDetector())
