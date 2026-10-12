import json
import shutil

import pandas as pd
import pytest
import torch

from deepfake_detector.model import load_checkpoint
from tests.conftest import write_video
from training.data import ClipAugment, Sample, VideoClipDataset, split_samples
from training.evaluate import PERTURBATIONS, perturb, summarise


def test_group_split_has_no_leakage():
    samples = [Sample(f"v{i}.mp4", i % 2, f"src{i // 4}") for i in range(40)]
    train, val, test = split_samples(samples, 0.2, 0.2, seed=0)
    groups = [{s.group for s in part} for part in (train, val, test)]
    assert not (groups[0] & groups[1]) and not (groups[0] & groups[2]) and not (groups[1] & groups[2])
    assert len(train) + len(val) + len(test) == 40


def test_dataset_pads_short_videos_and_augments(tmp_path):
    path = write_video(tmp_path / "short.mp4", frames=5, size=(64, 64))
    ds = VideoClipDataset([Sample(path, 0, "g")], sequence_length=8, image_size=32, augment=ClipAugment(p_jpeg=1.0))
    x, y = ds[0]
    assert x.shape == (8, 3, 32, 32) and y == 0
    torch.testing.assert_close(x[5], x[7])


def test_train_smoke(tmp_path):
    from training import train

    root = tmp_path / "faces"
    root.mkdir()
    rows = []
    for i in range(8):
        write_video(root / f"v{i}.mp4", frames=6, size=(48, 48))
        rows.append({"file": f"v{i}.mp4", "label": "FAKE" if i % 2 else "REAL"})
    pd.DataFrame(rows).to_csv(tmp_path / "meta.csv", index=False)

    out = tmp_path / "run"
    train.main(
        [
            "--metadata", str(tmp_path / "meta.csv"), "--video-root", str(root), "--output-dir", str(out),
            "--epochs", "1", "--batch-size", "2", "--sequence-length", "4", "--image-size", "32",
            "--num-workers", "0", "--no-pretrained", "--val-ratio", "0.25", "--test-ratio", "0.25", "--device", "cpu",
        ]
    )  # fmt: skip
    summary = json.loads((out / "summary.json").read_text())
    assert summary["best_epoch"] == 1 and "loss" in summary["test"]
    model, meta = load_checkpoint(out / "best.pt")
    assert model.spec.temporal_mode == "sequence" and meta["epoch"] == 1


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_perturbations_produce_readable_videos(video, tmp_path):
    from deepfake_detector.video import probe

    for name in PERTURBATIONS:
        out = perturb(video, name, tmp_path)
        assert probe(out).frame_count > 0


def test_summarise_handles_failures():
    df = pd.DataFrame(
        [
            {"perturbation": "none", "is_fake": 1, "p_fake": 0.9, "verdict": "fake", "latency_ms": 10, "status": "ok"},
            {"perturbation": "none", "is_fake": 0, "p_fake": 0.2, "verdict": "real", "latency_ms": 12, "status": "ok"},
            {"perturbation": "none", "is_fake": 0, "status": "no_face"},
        ]
    )
    s = summarise(df, 0.5)["none"]
    assert s["auc"] == 1.0 and s["coverage"] == pytest.approx(2 / 3)
