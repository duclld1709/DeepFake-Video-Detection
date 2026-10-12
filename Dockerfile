# CPU image serving the REST API (default) or the Gradio UI.
#
#   docker build -t deepfake-detector .
#   docker run -p 8000:8000 -v $PWD/weights:/app/weights deepfake-detector
#   docker run -p 7860:7860 -v $PWD/weights:/app/weights deepfake-detector serve-ui
#
# Build with --build-arg WITH_DLIB=false for a smaller image that uses the OpenCV face
# detector (faster build, less faithful to the training crops).

FROM python:3.11-slim

ARG WITH_DLIB=true
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && if [ "$WITH_DLIB" = "true" ]; then apt-get install -y --no-install-recommends build-essential cmake; fi \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY pyproject.toml README.md LICENSE ./
COPY deepfake_detector ./deepfake_detector
COPY training ./training
COPY scripts ./scripts
RUN pip install ".[api,ui]" \
    && if [ "$WITH_DLIB" = "true" ]; then pip install ".[dlib]"; fi

RUN useradd --create-home app && mkdir -p /app/weights && chown -R app /app
USER app

ENV DEEPFAKE_WEIGHTS_PATH=/app/weights/80_frame
EXPOSE 8000 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python scripts/healthcheck.py

ENTRYPOINT ["deepfake-detect"]
CMD ["serve-api", "--port", "8000"]
