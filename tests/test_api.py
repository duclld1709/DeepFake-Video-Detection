import pytest
from fastapi.testclient import TestClient

from deepfake_detector.api import create_app
from deepfake_detector.inference import DeepfakeAnalyzer
from tests.conftest import StubDetector


@pytest.fixture
def client(analyzer, config):
    with TestClient(create_app(lambda: analyzer, config)) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model"]["temporal_mode"] == "legacy"


def test_analyze_upload(client, video):
    with video.open("rb") as f:
        res = client.post("/v1/analyze?explain=2", files={"file": ("clip.mp4", f, "video/mp4")})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["verdict"] in {"fake", "real", "inconclusive"}
    assert len(body["explanations"]) == 2
    assert "X-Process-Time-Ms" in res.headers


def test_rejects_unsupported_type(client):
    res = client.post("/v1/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 415


def test_rejects_corrupt_video(client):
    res = client.post("/v1/analyze", files={"file": ("x.mp4", b"garbage", "video/mp4")})
    assert res.status_code == 400


def test_rejects_too_large(analyzer, config, video):
    config.max_upload_mb = 0
    with TestClient(create_app(lambda: analyzer, config)) as c, video.open("rb") as f:
        assert c.post("/v1/analyze", files={"file": ("clip.mp4", f, "video/mp4")}).status_code == 413


def test_no_face_is_422(legacy_model, config, video):
    analyzer = DeepfakeAnalyzer(legacy_model, config, face_detector=StubDetector(box=None))
    with TestClient(create_app(lambda: analyzer, config)) as c, video.open("rb") as f:
        assert c.post("/v1/analyze", files={"file": ("clip.mp4", f, "video/mp4")}).status_code == 422


def test_model_load_failure_is_visible(config):
    def broken():
        raise FileNotFoundError("weights missing")

    with TestClient(create_app(broken, config)) as c:
        health = c.get("/health")
        assert health.status_code == 503 and "weights missing" in health.json()["detail"]
        assert c.post("/v1/analyze", files={"file": ("a.mp4", b"x", "video/mp4")}).status_code == 503
