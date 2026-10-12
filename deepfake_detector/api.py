"""REST API.

Run with ``uvicorn deepfake_detector.api:app`` or ``deepfake-detect serve-api``.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from deepfake_detector import __version__
from deepfake_detector.config import DetectorConfig
from deepfake_detector.inference import DeepfakeAnalyzer, NoFaceDetectedError, VideoTooLongError
from deepfake_detector.schemas import AnalysisResult
from deepfake_detector.video import VideoReadError

logger = logging.getLogger(__name__)

ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_CHUNK = 1 << 20


def create_app(
    analyzer_factory: Callable[[], DeepfakeAnalyzer] | None = None,
    config: DetectorConfig | None = None,
) -> FastAPI:
    config = config or DetectorConfig.from_env()
    factory = analyzer_factory or (lambda: DeepfakeAnalyzer.from_config(config))
    state: dict = {"analyzer": None, "error": None}
    slots = threading.BoundedSemaphore(max(1, config.max_concurrency))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            state["analyzer"] = factory()
            logger.info("model loaded: %s", state["analyzer"].model_info)
        except Exception as exc:  # keep serving /health so the failure is visible
            logger.exception("failed to load model")
            state["error"] = str(exc)
        yield

    app = FastAPI(
        title="Deepfake Video Detection API",
        version=__version__,
        description="Upload a video, get a real/fake verdict with per-frame scores and Grad-CAM evidence.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def timing_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
        return response

    @app.get("/health")
    def health():
        analyzer: DeepfakeAnalyzer | None = state["analyzer"]
        if analyzer is None:
            return JSONResponse({"status": "unavailable", "detail": state["error"]}, status_code=503)
        return {"status": "ok", "model": analyzer.model_info.model_dump()}

    @app.post("/v1/analyze", response_model=AnalysisResult)
    def analyze(
        file: UploadFile = File(..., description="Video file (mp4, mov, avi, mkv, webm)."),
        num_clips: int | None = Query(None, ge=1, le=10, description="Clips sampled across the video."),
        explain: int = Query(0, ge=0, le=16, description="Number of Grad-CAM overlays to return."),
    ) -> AnalysisResult:
        analyzer: DeepfakeAnalyzer | None = state["analyzer"]
        if analyzer is None:
            raise HTTPException(503, f"model not loaded: {state['error']}")

        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(415, f"unsupported file type '{suffix}'; allowed: {sorted(ALLOWED_SUFFIXES)}")

        with tempfile.TemporaryDirectory(prefix="deepfake-") as tmp:
            path = Path(tmp) / f"upload{suffix}"
            _save_limited(file, path, config.max_upload_mb)

            if not slots.acquire(timeout=120):
                raise HTTPException(503, "server busy, retry later")
            try:
                return analyzer.analyze(path, num_clips=num_clips, explain_frames=explain)
            except NoFaceDetectedError as exc:
                raise HTTPException(422, str(exc)) from exc
            except VideoTooLongError as exc:
                raise HTTPException(413, str(exc)) from exc
            except VideoReadError as exc:
                raise HTTPException(400, str(exc)) from exc
            finally:
                slots.release()

    return app


def _save_limited(upload: UploadFile, dest: Path, max_mb: int) -> None:
    limit = max_mb * 1024 * 1024
    written = 0
    with dest.open("wb") as out:
        while chunk := upload.file.read(_CHUNK):
            written += len(chunk)
            if written > limit:
                raise HTTPException(413, f"file larger than {max_mb} MB")
            out.write(chunk)
    if written == 0:
        raise HTTPException(400, "empty file")
    upload.file.close()


def __getattr__(name: str):
    # Lazily build the default app so importing this module has no side effects.
    if name == "app":
        globals()["app"] = application = create_app()
        return application
    raise AttributeError(name)
