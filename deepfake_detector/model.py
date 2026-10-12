"""ResNeXt-50 + LSTM deepfake classifier.

Parameter names (``model.*``, ``lstm.*``, ``linear1.*``) are kept identical to the
original ``train.py`` so the published checkpoint loads without remapping.

Two temporal modes exist because the original implementation had a shape bug:

* ``legacy`` - The original code fed a ``(batch, seq, feat)`` tensor into an
  ``nn.LSTM`` built without ``batch_first=True``. At inference (batch = 1) the LSTM
  therefore saw a sequence of length 1 containing 80 independent "samples": every
  frame went through one LSTM step from a zero state, and the 80 outputs were
  averaged. The published checkpoint was trained in that regime, so this mode
  reproduces it exactly (for any batch size, without the cross-video leakage the
  bug caused during batch > 1 training). Because the head is linear, the clip logit
  equals the mean of per-frame logits, which gives us exact per-frame scores.
* ``sequence`` - The intended design: ``batch_first=True`` and a real recurrence over
  the frames of each clip. Use this for new training runs.

The original also passed ``bidirectional`` as the 4th positional argument of
``nn.LSTM``, which is ``bias``; the checkpoint therefore has a bias-free LSTM.
``lstm_bias`` defaults to ``False`` to stay compatible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models

from deepfake_detector.config import TemporalMode


@dataclass
class ModelSpec:
    num_classes: int = 2
    latent_dim: int = 2048
    hidden_dim: int = 2048
    lstm_layers: int = 1
    lstm_bias: bool = False
    bidirectional: bool = False
    dropout: float = 0.4
    temporal_mode: TemporalMode = "legacy"


class DeepfakeDetector(nn.Module):
    def __init__(self, spec: ModelSpec | None = None, *, pretrained_backbone: bool = True):
        super().__init__()
        self.spec = spec = spec or ModelSpec()
        if spec.temporal_mode not in ("legacy", "sequence"):
            raise ValueError(f"unknown temporal_mode: {spec.temporal_mode!r}")

        weights = models.ResNeXt50_32X4D_Weights.DEFAULT if pretrained_backbone else None
        backbone = models.resnext50_32x4d(weights=weights)
        self.model = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(
            spec.latent_dim,
            spec.hidden_dim,
            spec.lstm_layers,
            bias=spec.lstm_bias,
            batch_first=True,
            bidirectional=spec.bidirectional,
        )
        lstm_out = spec.hidden_dim * (2 if spec.bidirectional else 1)
        self.dp = nn.Dropout(spec.dropout)
        self.linear1 = nn.Linear(lstm_out, spec.num_classes)

    def encode_frames(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, T, C, H, W)`` -> feature maps ``(B*T, 2048, h, w)`` and pooled ``(B, T, 2048)``."""

        b, t, c, h, w = x.shape
        fmap = self.model(x.reshape(b * t, c, h, w))
        pooled = self.avgpool(fmap).reshape(b, t, -1)
        return fmap, pooled

    def temporal(self, pooled: torch.Tensor) -> torch.Tensor:
        """Pooled features ``(B, T, D)`` -> per-step hidden states ``(B, T, H)``."""

        b, t, d = pooled.shape
        if self.spec.temporal_mode == "legacy":
            # One LSTM step per frame from a zero state (see module docstring).
            out, _ = self.lstm(pooled.reshape(b * t, 1, d))
            return out.reshape(b, t, -1)
        out, _ = self.lstm(pooled)
        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(feature_maps, clip_logits)`` like the original model."""

        fmap, pooled = self.encode_frames(x)
        hidden = self.temporal(pooled)
        # Dropout belongs before the classifier; the original applied it to the logits.
        logits = self.linear1(self.dp(hidden.mean(dim=1)))
        return fmap, logits

    def frame_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Per-frame logits ``(B, T, num_classes)``. Only meaningful in ``legacy`` mode,
        where their mean equals the clip logit exactly."""

        if self.spec.temporal_mode != "legacy":
            raise RuntimeError("per-frame logits are only defined for temporal_mode='legacy'")
        _, pooled = self.encode_frames(x)
        return self.linear1(self.dp(self.temporal(pooled)))


def save_checkpoint(model: DeepfakeDetector, path: str | Path, **metadata: Any) -> None:
    """Save weights together with the spec needed to rebuild the model."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "deepfake-detector/v2",
            "spec": asdict(model.spec),
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    temporal_mode: TemporalMode | None = None,
) -> tuple[DeepfakeDetector, dict[str, Any]]:
    """Load either a v2 checkpoint or a raw ``state_dict`` from the original project.

    Raw state dicts are assumed to come from the original training script, i.e.
    ``temporal_mode='legacy'`` and a bias-free LSTM, unless overridden.
    """

    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if isinstance(payload, dict) and payload.get("format") == "deepfake-detector/v2":
        spec = ModelSpec(**payload["spec"])
        state_dict = payload["state_dict"]
        metadata = dict(payload.get("metadata") or {})
    else:
        state_dict = payload
        spec = ModelSpec(lstm_bias="lstm.bias_ih_l0" in state_dict)
        metadata = {"source": "legacy-state-dict"}

    if temporal_mode is not None:
        spec.temporal_mode = temporal_mode

    model = DeepfakeDetector(spec, pretrained_backbone=False)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, metadata
