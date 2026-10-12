"""Response models shared by the Python API, the REST API and the UI."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    FAKE = "fake"
    REAL = "real"
    INCONCLUSIVE = "inconclusive"


class VideoMeta(BaseModel):
    fps: float
    frame_count: int
    duration_s: float
    width: int
    height: int


class FrameScore(BaseModel):
    frame_index: int
    timestamp_s: float
    fake_probability: float | None = Field(
        None, description="Per-frame P(fake); only available for temporal_mode='legacy'."
    )
    box: list[int] = Field(description="Face box [top, right, bottom, left] in source pixels.")


class ClipResult(BaseModel):
    index: int
    start_s: float
    end_s: float
    frames_used: int
    fake_probability: float


class Explanation(BaseModel):
    frame_index: int
    timestamp_s: float
    fake_probability: float | None
    overlay_png_base64: str = Field(description="Grad-CAM heatmap blended over the face crop (PNG).")


class ModelInfo(BaseModel):
    version: str
    temporal_mode: str
    sequence_length: int
    face_backend: str
    device: str
    weights: str


class Timings(BaseModel):
    faces_ms: float
    inference_ms: float
    explain_ms: float
    total_ms: float


class AnalysisResult(BaseModel):
    verdict: Verdict
    fake_probability: float = Field(description="Mean P(fake) over analysed clips.")
    max_clip_fake_probability: float
    confidence: float = Field(description="P(predicted class); 0.5-1.0.")
    threshold: float
    clips: list[ClipResult]
    frames: list[FrameScore]
    explanations: list[Explanation] = []
    face_coverage: float = Field(description="Fraction of scanned frames in which a face was found.")
    video: VideoMeta
    model: ModelInfo
    warnings: list[str] = []
    timings: Timings
