"""End-to-end video analysis: sample clips -> track face -> classify -> explain."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from deepfake_detector import explain
from deepfake_detector.config import FAKE_INDEX, DetectorConfig
from deepfake_detector.faces import Box, FaceDetector, FaceTracker, build_face_detector
from deepfake_detector.model import DeepfakeDetector, load_checkpoint
from deepfake_detector.preprocessing import build_transform, crop, thumbnail_rgb, to_clip_tensor
from deepfake_detector.schemas import (
    AnalysisResult,
    ClipResult,
    Explanation,
    FrameScore,
    ModelInfo,
    Timings,
    Verdict,
    VideoMeta,
)
from deepfake_detector.video import VideoInfo, VideoReadError, iter_frames, plan_clips, probe

logger = logging.getLogger(__name__)

__all__ = ["DeepfakeAnalyzer", "NoFaceDetectedError", "VideoReadError", "VideoTooLongError"]


class NoFaceDetectedError(ValueError):
    """Not enough frames with a detectable face to make a prediction."""


class VideoTooLongError(ValueError):
    """The video exceeds the configured maximum duration."""


@dataclass
class _Clip:
    index: int
    start: int
    stop: int
    frame_indices: list[int] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)
    tensors: list[torch.Tensor] = field(default_factory=list)
    thumbs: list[np.ndarray] = field(default_factory=list)
    scanned: int = 0


class DeepfakeAnalyzer:
    """Load once, analyse many videos. Stateless between calls apart from the model."""

    def __init__(
        self,
        model: DeepfakeDetector,
        config: DetectorConfig | None = None,
        *,
        face_detector: FaceDetector | None = None,
        weights_label: str = "in-memory",
    ):
        self.config = config or DetectorConfig()
        self.device = torch.device(self.config.resolve_device())
        self.model = model.to(self.device).eval()
        self.face_detector = face_detector or build_face_detector(self.config.face_backend, self.config.face_max_side)
        self.transform = build_transform(self.config.image_size)
        self.weights_label = weights_label

    @classmethod
    def from_config(cls, config: DetectorConfig | None = None) -> DeepfakeAnalyzer:
        config = config or DetectorConfig.from_env()
        path = Path(config.weights_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"model weights not found at '{path}'. Run `python scripts/download_weights.py` "
                "or set DEEPFAKE_WEIGHTS_PATH."
            )
        model, _ = load_checkpoint(path, device=config.resolve_device(), temporal_mode=config.temporal_mode)
        return cls(model, config, weights_label=path.name)

    @property
    def model_info(self) -> ModelInfo:
        from deepfake_detector import __version__

        return ModelInfo(
            version=__version__,
            temporal_mode=self.model.spec.temporal_mode,
            sequence_length=self.config.sequence_length,
            face_backend=self.face_detector.name,
            device=str(self.device),
            weights=self.weights_label,
        )

    # ------------------------------------------------------------------ public API

    def analyze(
        self,
        video_path: str | Path,
        *,
        num_clips: int | None = None,
        explain_frames: int | None = 0,
    ) -> AnalysisResult:
        """Analyse a video file.

        ``explain_frames`` - number of Grad-CAM overlays to return (``None`` -> config default,
        ``0`` -> none). They are taken from the most suspicious frames.
        """

        cfg = self.config
        t0 = time.perf_counter()
        info = probe(video_path)
        if info.duration_s > cfg.max_duration_s:
            raise VideoTooLongError(f"video is {info.duration_s:.0f}s long; the limit is {cfg.max_duration_s:.0f}s")

        warnings: list[str] = []
        if self.face_detector.name != "face_recognition":
            warnings.append(
                f"Face backend '{self.face_detector.name}' differs from the dlib detector used in training; "
                "scores may be less reliable."
            )

        clips, scanned = self._collect_faces(video_path, info, num_clips or cfg.num_clips, warnings)
        t_faces = time.perf_counter()

        clip_results, frame_scores, clip_probs, pooled_by_clip = self._classify(clips, info)
        t_infer = time.perf_counter()

        k = cfg.gradcam_frames if explain_frames is None else explain_frames
        explanations = self._explain(clips, frame_scores, clip_probs, pooled_by_clip, info, k) if k > 0 else []
        t_explain = time.perf_counter()

        p_fake = float(np.mean(clip_probs))
        verdict = self._verdict(p_fake)
        used = sum(len(c.frame_indices) for c in clips)
        return AnalysisResult(
            verdict=verdict,
            fake_probability=round(p_fake, 4),
            max_clip_fake_probability=round(float(np.max(clip_probs)), 4),
            confidence=round(p_fake if p_fake >= cfg.threshold else 1 - p_fake, 4),
            threshold=cfg.threshold,
            clips=clip_results,
            frames=frame_scores,
            explanations=explanations,
            face_coverage=round(used / scanned, 4) if scanned else 0.0,
            video=VideoMeta(
                fps=info.fps,
                frame_count=info.frame_count,
                duration_s=round(info.duration_s, 3),
                width=info.width,
                height=info.height,
            ),
            model=self.model_info,
            warnings=warnings,
            timings=Timings(
                faces_ms=_ms(t0, t_faces),
                inference_ms=_ms(t_faces, t_infer),
                explain_ms=_ms(t_infer, t_explain),
                total_ms=_ms(t0, t_explain),
            ),
        )

    # ------------------------------------------------------------------ stages

    def _collect_faces(
        self, video_path: str | Path, info: VideoInfo, num_clips: int, warnings: list[str]
    ) -> tuple[list[_Clip], int]:
        cfg = self.config
        seq_len = cfg.sequence_length
        min_faces = max(1, int(np.ceil(cfg.min_face_ratio * seq_len)))
        tracker = FaceTracker(self.face_detector, detect_every=cfg.detect_every)

        clips: list[_Clip] = []
        scanned_total = 0
        for i, (start, stop) in enumerate(plan_clips(info.frame_count, num_clips, seq_len)):
            clip = _Clip(index=i, start=start, stop=stop)
            tracker.reset()
            for frame_index, frame in iter_frames(video_path, start, stop):
                clip.scanned += 1
                box = tracker.update(frame)
                if box is None:
                    continue
                face = crop(frame, box)
                if face.size == 0:
                    continue
                clip.frame_indices.append(frame_index)
                clip.boxes.append(box)
                clip.tensors.append(self.transform(face))
                clip.thumbs.append(thumbnail_rgb(face, cfg.image_size))
                if len(clip.tensors) == seq_len:
                    break
            scanned_total += clip.scanned

            if len(clip.tensors) < min_faces:
                warnings.append(
                    f"Clip {i} ({start / info.fps:.1f}s-{stop / info.fps:.1f}s) skipped: "
                    f"face found in {len(clip.tensors)} frames, need {min_faces}."
                )
                continue
            if len(clip.tensors) < seq_len:
                warnings.append(f"Clip {i} used {len(clip.tensors)}/{seq_len} frames.")
            clips.append(clip)

        if not clips:
            raise NoFaceDetectedError(
                "No clip contained enough frames with a detectable face. Use a video where one face is clearly visible."
            )
        return clips, scanned_total

    @torch.no_grad()  # not inference_mode: pooled features are reused by Grad-CAM
    def _classify(self, clips: list[_Clip], info: VideoInfo):
        legacy = self.model.spec.temporal_mode == "legacy"
        clip_results: list[ClipResult] = []
        frame_scores: list[FrameScore] = []
        clip_probs: list[float] = []
        pooled_by_clip: list[torch.Tensor] = []

        for clip in clips:
            x = to_clip_tensor(clip.tensors).to(self.device)
            _, pooled = self.model.encode_frames(x)
            hidden = self.model.temporal(pooled)
            clip_logits = self.model.linear1(hidden.mean(dim=1))
            p_clip = torch.softmax(clip_logits, dim=-1)[0, FAKE_INDEX].item()
            per_frame = (
                torch.softmax(self.model.linear1(hidden), dim=-1)[0, :, FAKE_INDEX].cpu().tolist()
                if legacy
                else [None] * len(clip.frame_indices)
            )

            pooled_by_clip.append(pooled)
            clip_probs.append(p_clip)
            clip_results.append(
                ClipResult(
                    index=clip.index,
                    start_s=round(clip.frame_indices[0] / info.fps, 3),
                    end_s=round(clip.frame_indices[-1] / info.fps, 3),
                    frames_used=len(clip.frame_indices),
                    fake_probability=round(p_clip, 4),
                )
            )
            frame_scores.extend(
                FrameScore(
                    frame_index=fi,
                    timestamp_s=round(fi / info.fps, 3),
                    fake_probability=None if p is None else round(p, 4),
                    box=box.as_list(),
                )
                for fi, p, box in zip(clip.frame_indices, per_frame, clip.boxes, strict=True)
            )
        return clip_results, frame_scores, clip_probs, pooled_by_clip

    def _explain(self, clips, frame_scores, clip_probs, pooled_by_clip, info: VideoInfo, k: int) -> list[Explanation]:
        # Rank (clip, position) pairs: by per-frame score in legacy mode, otherwise
        # evenly spaced frames from the most suspicious clip.
        candidates: list[tuple[float, int, int]] = []
        offset = 0
        for ci, clip in enumerate(clips):
            n = len(clip.frame_indices)
            scores = [frame_scores[offset + j].fake_probability for j in range(n)]
            offset += n
            if scores and scores[0] is not None:
                candidates.extend((s, ci, j) for j, s in enumerate(scores))
        if candidates:
            chosen = sorted(candidates, reverse=True)[:k]
        else:
            ci = int(np.argmax(clip_probs))
            n = len(clips[ci].frame_indices)
            positions = sorted({int(p) for p in np.linspace(0, n - 1, min(k, n))})
            chosen = [(clip_probs[ci], ci, j) for j in positions]

        by_clip: dict[int, list[tuple[float, int]]] = {}
        for score, ci, j in chosen:
            by_clip.setdefault(ci, []).append((score, j))

        out: list[Explanation] = []
        for ci, items in by_clip.items():
            clip = clips[ci]
            positions = [j for _, j in items]
            x = to_clip_tensor(clip.tensors).to(self.device)
            heatmaps = explain.gradcam(self.model, x, positions, pooled=pooled_by_clip[ci])
            for (score, j), heat in zip(items, heatmaps, strict=True):
                fi = clip.frame_indices[j]
                per_frame = score if candidates else None
                out.append(
                    Explanation(
                        frame_index=fi,
                        timestamp_s=round(fi / info.fps, 3),
                        fake_probability=None if per_frame is None else round(per_frame, 4),
                        overlay_png_base64=explain.png_base64(explain.overlay(clip.thumbs[j], heat)),
                    )
                )
        out.sort(key=lambda e: e.frame_index)
        return out

    def _verdict(self, p_fake: float) -> Verdict:
        cfg = self.config
        if abs(p_fake - cfg.threshold) < cfg.inconclusive_margin:
            return Verdict.INCONCLUSIVE
        return Verdict.FAKE if p_fake >= cfg.threshold else Verdict.REAL


def _ms(start: float, end: float) -> float:
    return round((end - start) * 1000, 1)
