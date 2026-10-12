import base64

import cv2
import numpy as np
import pytest
import torch

from deepfake_detector.inference import DeepfakeAnalyzer, NoFaceDetectedError, VideoTooLongError
from deepfake_detector.metrics import binary_report, equal_error_rate, expected_calibration_error
from deepfake_detector.schemas import Verdict
from tests.conftest import StubDetector


def test_analyze_returns_consistent_result(analyzer, video):
    result = analyzer.analyze(video, explain_frames=0)
    assert len(result.clips) == 2
    assert all(c.frames_used == 8 for c in result.clips)
    assert len(result.frames) == 16
    assert 0.0 <= result.fake_probability <= 1.0
    assert result.fake_probability == pytest.approx(np.mean([c.fake_probability for c in result.clips]), abs=1e-3)
    # In legacy mode each clip score is the softmax of the mean per-frame logit, so it lies
    # within the range of its per-frame scores.
    for clip in result.clips:
        frames = [f.fake_probability for f in result.frames if clip.start_s <= f.timestamp_s <= clip.end_s]
        assert min(frames) - 1e-3 <= clip.fake_probability <= max(frames) + 1e-3
    assert result.face_coverage == 1.0
    assert result.explanations == []
    assert any("differs from the dlib detector" in w for w in result.warnings)


def test_analyze_is_deterministic(analyzer, video):
    a = analyzer.analyze(video).model_dump(exclude={"timings"})
    b = analyzer.analyze(video).model_dump(exclude={"timings"})
    assert a == b


def test_explanations_are_valid_pngs(analyzer, video):
    result = analyzer.analyze(video, explain_frames=3)
    assert len(result.explanations) == 3
    top = sorted((f.fake_probability for f in result.frames), reverse=True)[:3]
    assert sorted((e.fake_probability for e in result.explanations), reverse=True) == top
    img = cv2.imdecode(np.frombuffer(base64.b64decode(result.explanations[0].overlay_png_base64), np.uint8), 1)
    assert img.shape == (112, 112, 3)


def test_gradcam_in_sequence_mode(video, config):
    from deepfake_detector.model import DeepfakeDetector, ModelSpec

    model = DeepfakeDetector(ModelSpec(temporal_mode="sequence", hidden_dim=32), pretrained_backbone=False)
    result = DeepfakeAnalyzer(model, config, face_detector=StubDetector()).analyze(video, explain_frames=2)
    assert all(f.fake_probability is None for f in result.frames)
    assert len(result.explanations) == 2


def test_partial_face_coverage_is_reported(legacy_model, config, video):
    analyzer = DeepfakeAnalyzer(legacy_model, config, face_detector=StubDetector(every=2))
    result = analyzer.analyze(video)
    assert 0.4 < result.face_coverage < 0.6


def test_no_face_raises(legacy_model, config, video):
    analyzer = DeepfakeAnalyzer(legacy_model, config, face_detector=StubDetector(box=None))
    with pytest.raises(NoFaceDetectedError):
        analyzer.analyze(video)


def test_too_long_raises(legacy_model, config, video):
    config.max_duration_s = 1.0
    with pytest.raises(VideoTooLongError):
        DeepfakeAnalyzer(legacy_model, config, face_detector=StubDetector()).analyze(video)


@pytest.mark.parametrize("p,expected", [(0.9, Verdict.FAKE), (0.1, Verdict.REAL), (0.55, Verdict.INCONCLUSIVE)])
def test_verdict_thresholds(analyzer, p, expected):
    assert analyzer._verdict(p) == expected


def test_missing_weights_message(tmp_path, config):
    config.weights_path = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="download_weights"):
        DeepfakeAnalyzer.from_config(config)


def test_metrics():
    y = [0, 0, 1, 1]
    report = binary_report(y, [0.1, 0.2, 0.8, 0.9])
    assert report["auc"] == 1.0 and report["accuracy"] == 1.0 and report["f1"] == 1.0
    eer, _ = equal_error_rate(y, [0.1, 0.9, 0.2, 0.8])
    assert eer == pytest.approx(0.5)
    assert expected_calibration_error([1, 1, 0, 0], [1.0, 1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert expected_calibration_error([0, 0], [0.9, 0.9]) == pytest.approx(0.9)
    assert "auc" not in binary_report([1, 1], [0.6, 0.7])


def test_device_is_cpu(analyzer):
    assert analyzer.device == torch.device("cpu")
