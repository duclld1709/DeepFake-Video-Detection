"""Command line entry point: ``deepfake-detect {analyze,serve-api,serve-ui}``."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
from pathlib import Path

from deepfake_detector.config import DetectorConfig


def _config(args: argparse.Namespace) -> DetectorConfig:
    return DetectorConfig.from_env(
        weights_path=Path(args.weights) if args.weights else None,
        device=args.device,
        face_backend=args.face_backend,
    )


def cmd_analyze(args: argparse.Namespace) -> int:
    from deepfake_detector.inference import DeepfakeAnalyzer, NoFaceDetectedError

    analyzer = DeepfakeAnalyzer.from_config(_config(args))
    explain = args.explain if args.explain_dir else 0
    try:
        result = analyzer.analyze(args.video, num_clips=args.clips, explain_frames=explain)
    except NoFaceDetectedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.explain_dir:
        out = Path(args.explain_dir)
        out.mkdir(parents=True, exist_ok=True)
        for e in result.explanations:
            (out / f"frame_{e.frame_index:06d}.png").write_bytes(base64.b64decode(e.overlay_png_base64))

    if args.json:
        print(json.dumps(result.model_dump(mode="json", exclude={"explanations"}), indent=2))
    else:
        print(f"verdict:      {result.verdict.value}")
        print(f"P(fake):      {result.fake_probability:.4f}  (threshold {result.threshold})")
        print(f"clips:        {', '.join(f'{c.fake_probability:.3f}' for c in result.clips)}")
        print(f"face coverage {result.face_coverage:.0%}, total {result.timings.total_ms:.0f} ms")
        for w in result.warnings:
            print(f"note: {w}")
    return 0


def cmd_serve_api(args: argparse.Namespace) -> int:
    import uvicorn

    from deepfake_detector.api import create_app

    uvicorn.run(create_app(config=_config(args)), host=args.host, port=args.port)
    return 0


def cmd_serve_ui(args: argparse.Namespace) -> int:
    from deepfake_detector.inference import DeepfakeAnalyzer
    from deepfake_detector.ui import build_ui

    build_ui(DeepfakeAnalyzer.from_config(_config(args))).launch(
        server_name=args.host, server_port=args.port, share=args.share
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deepfake-detect", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--weights", help="checkpoint path (default: $DEEPFAKE_WEIGHTS_PATH or weights/80_frame)")
    common.add_argument("--device", default=None, help="cpu, cuda, cuda:0 ... (default: auto)")
    common.add_argument("--face-backend", choices=["auto", "face_recognition", "opencv"], default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", parents=[common], help="analyse one video")
    p.add_argument("video")
    p.add_argument("--clips", type=int, default=None)
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    p.add_argument("--explain", type=int, default=6, help="Grad-CAM frames to save (with --explain-dir)")
    p.add_argument("--explain-dir", help="directory for Grad-CAM PNGs")
    p.set_defaults(func=cmd_analyze)

    for name, func, port in (("serve-api", cmd_serve_api, 8000), ("serve-ui", cmd_serve_ui, 7860)):
        s = sub.add_parser(name, parents=[common])
        s.add_argument("--host", default="0.0.0.0")
        s.add_argument("--port", type=int, default=port)
        if name == "serve-ui":
            s.add_argument("--share", action="store_true", help="create a public Gradio link")
        s.set_defaults(func=func)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
