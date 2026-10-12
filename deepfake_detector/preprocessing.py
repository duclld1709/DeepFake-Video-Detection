"""Face crop -> model tensor.

The checkpoint was trained on frames read with OpenCV and fed to ``ToPILImage``
without colour conversion, i.e. in **BGR** order with ImageNet statistics. We keep
that order on purpose: "fixing" it to RGB would silently shift the input
distribution the weights were trained on.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torchvision import transforms

from deepfake_detector.config import IMAGENET_MEAN, IMAGENET_STD
from deepfake_detector.faces import Box


def build_transform(image_size: int = 112) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def crop(frame_bgr: np.ndarray, box: Box) -> np.ndarray:
    return frame_bgr[box.top : box.bottom, box.left : box.right]


def thumbnail_rgb(face_bgr: np.ndarray, size: int = 112) -> np.ndarray:
    """Display copy of a face crop (RGB, ``size`` x ``size``)."""

    return cv2.cvtColor(cv2.resize(face_bgr, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def to_clip_tensor(faces: list[torch.Tensor]) -> torch.Tensor:
    """List of ``(C, H, W)`` tensors -> ``(1, T, C, H, W)``."""

    return torch.stack(faces).unsqueeze(0)
