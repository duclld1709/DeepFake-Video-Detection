"""Dataset and leakage-free splitting for face-cropped training videos.

Metadata CSV columns:

* ``file``  - path of a face-cropped video, relative to ``--video-root``
* ``label`` - ``FAKE`` or ``REAL``
* ``group`` - optional; videos sharing a group never cross splits. Use the source
  (original) video or the identity of the person. With DFDC, a fake's ``original``
  field gives its source video, so ``group = original or file``.

Without a group column the split falls back to a per-file stratified split and a
warning is printed: fakes generated from the same source then leak across splits,
which inflates validation accuracy (this is how the original 90% was measured).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import Dataset

from deepfake_detector.preprocessing import build_transform

logger = logging.getLogger(__name__)

LABELS = {"FAKE": 0, "REAL": 1}  # same class order as the original checkpoint


@dataclass
class Sample:
    path: Path
    label: int
    group: str


def load_samples(metadata_csv: str | Path, video_root: str | Path) -> list[Sample]:
    df = pd.read_csv(metadata_csv)
    if "file" not in df or "label" not in df:
        raise ValueError("metadata CSV needs 'file' and 'label' columns")
    root = Path(video_root)
    has_group = "group" in df and df["group"].notna().any()
    if not has_group:
        logger.warning("no 'group' column: splitting per file; source-video leakage across splits is possible")
    samples = []
    for row in df.itertuples(index=False):
        label = str(row.label).strip().upper()
        if label not in LABELS:
            raise ValueError(f"unknown label {row.label!r} for {row.file}")
        group = str(row.group) if has_group and pd.notna(row.group) else str(row.file)
        samples.append(Sample(root / str(row.file), LABELS[label], group))
    return samples


def split_samples(
    samples: list[Sample], val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Group-aware train/val/test split (stratified per file when groups are unique)."""

    idx = np.arange(len(samples))
    groups = np.array([s.group for s in samples])
    labels = np.array([s.label for s in samples])

    def _split(indices: np.ndarray, ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
        if ratio <= 0:
            return indices, indices[:0]
        if len(np.unique(groups[indices])) == len(indices):
            return train_test_split(indices, test_size=ratio, random_state=seed, stratify=labels[indices])
        gss = GroupShuffleSplit(n_splits=1, test_size=ratio, random_state=seed)
        a, b = next(gss.split(indices, groups=groups[indices]))
        return indices[a], indices[b]

    rest, test = _split(idx, test_ratio, seed)
    train, val = _split(rest, val_ratio / (1 - test_ratio), seed + 1)
    pick = lambda ids: [samples[i] for i in ids]  # noqa: E731
    return pick(train), pick(val), pick(test)


class ClipAugment:
    """Augmentations applied identically to every frame of a clip.

    Recompression and blur matter most: real-world deepfakes are re-encoded by
    messaging apps and social networks, which destroys the high-frequency artefacts
    a detector trained on pristine files leans on.
    """

    def __init__(self, p_flip=0.5, p_jpeg=0.5, jpeg_quality=(35, 90), p_blur=0.2, p_color=0.5):
        self.p_flip, self.p_jpeg, self.jpeg_quality = p_flip, p_jpeg, jpeg_quality
        self.p_blur, self.p_color = p_blur, p_color

    def __call__(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        flip = random.random() < self.p_flip
        quality = random.randint(*self.jpeg_quality) if random.random() < self.p_jpeg else None
        blur = random.choice([3, 5]) if random.random() < self.p_blur else None
        gain, bias = (random.uniform(0.8, 1.2), random.uniform(-20, 20)) if random.random() < self.p_color else (1, 0)
        out = []
        for f in frames:
            if flip:
                f = cv2.flip(f, 1)
            if gain != 1 or bias != 0:
                f = cv2.convertScaleAbs(f, alpha=gain, beta=bias)
            if blur:
                f = cv2.GaussianBlur(f, (blur, blur), 0)
            if quality:
                f = cv2.imdecode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], cv2.IMREAD_COLOR)
            out.append(f)
        return out


class VideoClipDataset(Dataset):
    """Reads ``sequence_length`` consecutive frames from a face-cropped video.

    Frames stay in BGR order (OpenCV) to match the published checkpoint. Videos shorter
    than the clip are padded by repeating the last frame instead of crashing.
    """

    def __init__(
        self,
        samples: list[Sample],
        sequence_length: int = 80,
        image_size: int = 112,
        augment: ClipAugment | None = None,
        random_start: bool = False,
    ):
        self.samples = samples
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.augment = augment
        self.random_start = random_start
        # Same transform as inference, so training and serving see identical inputs.
        self.transform = build_transform(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[i]
        frames = self._read(sample.path)
        if self.augment:
            frames = self.augment(frames)
        return torch.stack([self.transform(f) for f in frames]), sample.label

    def _read(self, path: Path) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            start = 0
            if self.random_start and total > self.sequence_length:
                start = random.randint(0, total - self.sequence_length)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames = []
            while len(frames) < self.sequence_length:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            cap.release()
        if not frames:
            raise RuntimeError(f"no frames decoded from {path}")
        frames += [frames[-1]] * (self.sequence_length - len(frames))
        return frames
