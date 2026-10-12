"""Grad-CAM over the ResNeXt feature maps.

Only the selected frames are re-encoded with gradients; the remaining frames reuse
their (constant) pooled features, so gradients for the selected frames are exact
while memory stays proportional to the number of explained frames.
"""

from __future__ import annotations

import base64
import contextlib

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from deepfake_detector.config import FAKE_INDEX
from deepfake_detector.model import DeepfakeDetector


def gradcam(
    model: DeepfakeDetector,
    clip: torch.Tensor,
    frame_indices: list[int],
    pooled: torch.Tensor | None = None,
) -> np.ndarray:
    """Return heatmaps ``(k, H, W)`` in ``[0, 1]`` for ``frame_indices`` of ``clip`` ``(1, T, C, H, W)``.

    The target is the fake-vs-real log-odds: per frame in ``legacy`` mode, for the
    whole clip in ``sequence`` mode.
    """

    if not frame_indices:
        return np.zeros((0, clip.shape[-2], clip.shape[-1]), dtype=np.float32)

    model.eval()
    idx = torch.as_tensor(frame_indices, device=clip.device)
    real_index = 1 - FAKE_INDEX
    cudnn_off = torch.backends.cudnn.flags(enabled=False) if clip.is_cuda else contextlib.nullcontext()

    with cudnn_off, torch.enable_grad():  # cuDNN RNN backward is not allowed in eval mode
        if pooled is None:
            with torch.no_grad():
                _, pooled = model.encode_frames(clip)
        fmap, pooled_sel = model.encode_frames(clip[:, idx])
        fmap.retain_grad()
        pooled_all = pooled.detach().clone()
        pooled_all[:, idx] = pooled_sel

        hidden = model.temporal(pooled_all)
        if model.spec.temporal_mode == "legacy":
            logits = model.linear1(hidden[:, idx])  # (1, k, 2)
            target = (logits[..., FAKE_INDEX] - logits[..., real_index]).sum()
        else:
            logits = model.linear1(hidden.mean(dim=1))
            target = (logits[:, FAKE_INDEX] - logits[:, real_index]).sum()
        target.backward()

    grads, acts = fmap.grad.detach(), fmap.detach()  # (k, 2048, h, w)
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=clip.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    flat = cam.flatten(1)
    lo, hi = flat.min(dim=1).values[:, None, None], flat.max(dim=1).values[:, None, None]
    cam = (cam - lo) / (hi - lo).clamp_min(1e-8)
    model.zero_grad(set_to_none=True)
    return cam.cpu().numpy().astype(np.float32)


def overlay(face_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a ``[0, 1]`` heatmap (JET colormap) onto an RGB image of any size."""

    h, w = face_rgb.shape[:2]
    heat = cv2.resize((heatmap * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
    color = cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(face_rgb, 1 - alpha, color, alpha, 0)


def png_base64(image_rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")
