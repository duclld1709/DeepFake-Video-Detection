import pytest
import torch
import torch.nn as nn
from torchvision import models

from deepfake_detector.model import DeepfakeDetector, ModelSpec, load_checkpoint, save_checkpoint


class OriginalModel(nn.Module):
    """Verbatim copy of the v1 model (minus the pretrained download) to pin compatibility."""

    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super().__init__()
        model = models.resnext50_32x4d(weights=None)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(torch.mean(x_lstm, dim=1)))


@pytest.fixture(scope="module")
def original():
    torch.manual_seed(1)
    return OriginalModel(2).eval()


def test_original_lstm_has_no_bias(original):
    # nn.LSTM's 4th positional argument is `bias`, so `bidirectional=False` disabled it.
    assert "lstm.bias_ih_l0" not in original.state_dict()


def test_legacy_mode_reproduces_original_inference(original, tmp_path):
    path = tmp_path / "80_frame"
    torch.save(original.state_dict(), path)
    model, meta = load_checkpoint(path)
    assert meta["source"] == "legacy-state-dict"
    assert model.spec.temporal_mode == "legacy" and model.spec.lstm_bias is False

    x = torch.randn(1, 6, 3, 64, 64)
    with torch.no_grad():
        _, expected = original(x)
        _, actual = model(x)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_original_leaks_across_batch_but_legacy_does_not(original, tmp_path):
    path = tmp_path / "w"
    torch.save(original.state_dict(), path)
    model, _ = load_checkpoint(path)
    a, b = torch.randn(1, 4, 3, 64, 64), torch.randn(1, 4, 3, 64, 64)
    with torch.no_grad():
        b_alone = model(b)[1]
        b_batched = model(torch.cat([a, b]))[1][1:]
        orig_b_alone = original(b)[1]
        orig_b_batched = original(torch.cat([a, b]))[1][1:]
    torch.testing.assert_close(b_batched, b_alone, atol=1e-5, rtol=1e-4)
    # Without batch_first the original LSTM treats the batch axis as time, so the second
    # video's prediction depends on the hidden state left by the first one.
    assert not torch.allclose(orig_b_batched, orig_b_alone, atol=1e-4)


def test_frame_logits_average_to_clip_logits(legacy_model):
    x = torch.randn(2, 5, 3, 64, 64)
    with torch.no_grad():
        _, clip = legacy_model(x)
        frames = legacy_model.frame_logits(x)
    torch.testing.assert_close(frames.mean(dim=1), clip, atol=1e-5, rtol=1e-4)


def test_sequence_mode_runs_and_has_no_frame_logits():
    model = DeepfakeDetector(ModelSpec(temporal_mode="sequence", hidden_dim=64), pretrained_backbone=False).eval()
    with torch.no_grad():
        fmap, logits = model(torch.randn(2, 3, 3, 64, 64))
    assert logits.shape == (2, 2) and fmap.shape[0] == 6
    with pytest.raises(RuntimeError):
        model.frame_logits(torch.randn(1, 3, 3, 64, 64))


def test_v2_checkpoint_roundtrip(tmp_path):
    spec = ModelSpec(temporal_mode="sequence", hidden_dim=32, lstm_bias=True)
    model = DeepfakeDetector(spec, pretrained_backbone=False)
    save_checkpoint(model, tmp_path / "best.pt", epoch=3)
    loaded, meta = load_checkpoint(tmp_path / "best.pt")
    assert loaded.spec == spec and meta == {"epoch": 3}
    for k, v in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[k], v)


def test_unknown_temporal_mode_rejected():
    with pytest.raises(ValueError):
        DeepfakeDetector(ModelSpec(temporal_mode="bogus"), pretrained_backbone=False)
