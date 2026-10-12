"""Gradio demo entry point (also used by Hugging Face Spaces).

python app.py            # http://localhost:7860
"""

import os

from deepfake_detector.config import DetectorConfig
from deepfake_detector.inference import DeepfakeAnalyzer
from deepfake_detector.ui import build_ui

if __name__ == "__main__":
    analyzer = DeepfakeAnalyzer.from_config(DetectorConfig.from_env())
    build_ui(analyzer).launch(
        server_name=os.environ.get("HOST", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", "7860")),
        share=os.environ.get("GRADIO_SHARE", "").lower() in {"1", "true"},
    )
