"""Deepfake video detection: ResNeXt-50 frame encoder + LSTM temporal head."""

from deepfake_detector.config import DetectorConfig
from deepfake_detector.inference import DeepfakeAnalyzer, NoFaceDetectedError, VideoReadError
from deepfake_detector.model import DeepfakeDetector, load_checkpoint, save_checkpoint
from deepfake_detector.schemas import AnalysisResult, Verdict

__version__ = "2.0.0"

__all__ = [
    "AnalysisResult",
    "DeepfakeAnalyzer",
    "DeepfakeDetector",
    "DetectorConfig",
    "NoFaceDetectedError",
    "Verdict",
    "VideoReadError",
    "__version__",
    "load_checkpoint",
    "save_checkpoint",
]
