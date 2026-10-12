"""Runtime configuration, overridable through ``DEEPFAKE_*`` environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Literal

# Class order used by the original training script: label 0 = FAKE, 1 = REAL.
CLASS_NAMES: tuple[str, str] = ("FAKE", "REAL")
FAKE_INDEX = 0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

FaceBackend = Literal["auto", "face_recognition", "opencv"]
TemporalMode = Literal["legacy", "sequence"]


@dataclass
class DetectorConfig:
    """Everything the inference pipeline needs to know.

    Defaults reproduce the setup the published ``80_frame`` checkpoint was trained with:
    80 consecutive face crops, 112x112, BGR channel order, ImageNet normalisation.
    """

    weights_path: Path = Path("weights/80_frame")
    device: str = "auto"

    # Clip sampling
    sequence_length: int = 80
    num_clips: int = 3
    image_size: int = 112
    min_face_ratio: float = 0.25  # fraction of sequence_length that must contain a face

    # Face detection
    face_backend: FaceBackend = "auto"
    face_max_side: int = 640  # frames are downscaled to this before detection (speed)
    detect_every: int = 1  # run the detector every N frames, reuse the box in between

    # Decision
    threshold: float = 0.5  # P(fake) >= threshold -> FAKE
    inconclusive_margin: float = 0.1  # |P(fake) - threshold| < margin -> INCONCLUSIVE

    # Explainability
    gradcam_frames: int = 6

    # Service limits
    max_upload_mb: int = 200
    max_duration_s: float = 600.0
    max_concurrency: int = 1

    # Model overrides; None means "read from checkpoint metadata".
    temporal_mode: TemporalMode | None = None

    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, prefix: str = "DEEPFAKE_", **overrides) -> DetectorConfig:
        """Build a config from defaults, then environment variables, then explicit overrides."""

        values: dict = {}
        for f in fields(cls):
            if f.name == "extra":
                continue
            raw = os.environ.get(prefix + f.name.upper())
            if raw is None:
                continue
            values[f.name] = _coerce(f.name, raw, getattr(cls(), f.name))
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"


def _coerce(name: str, raw: str, default):
    if name == "weights_path":
        return Path(raw)
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw
