"""End-to-end evaluation on raw videos, including robustness to real-world degradations.

    python -m training.evaluate --metadata data/celebdf_test.csv --video-root data/celebdf \
        --weights weights/80_frame --output-dir reports/celebdf \
        --perturbations none,h264_crf28,h264_crf35,downscale_0.5,fps_15

This runs the exact pipeline users hit (face detection + tracking + model), so the
numbers describe the product, not just the classifier. Use a dataset that was *not*
used for training (e.g. Celeb-DF or DFDC when trained on FaceForensics++) to measure
cross-dataset generalisation, the main failure mode of deepfake detectors.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd

from deepfake_detector.config import DetectorConfig
from deepfake_detector.inference import DeepfakeAnalyzer, NoFaceDetectedError
from deepfake_detector.metrics import binary_report

logger = logging.getLogger("evaluate")

# name -> ffmpeg arguments (between input and output)
PERTURBATIONS: dict[str, list[str]] = {
    "none": [],
    "h264_crf23": ["-c:v", "libx264", "-crf", "23", "-preset", "veryfast"],
    "h264_crf28": ["-c:v", "libx264", "-crf", "28", "-preset", "veryfast"],
    "h264_crf35": ["-c:v", "libx264", "-crf", "35", "-preset", "veryfast"],
    "downscale_0.5": [
        "-vf",
        "scale=trunc(iw/4)*2:trunc(ih/4)*2",
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "veryfast",
    ],
    "fps_15": ["-vf", "fps=15", "-c:v", "libx264", "-crf", "23", "-preset", "veryfast"],
}


def perturb(src: Path, name: str, workdir: Path) -> Path:
    args = PERTURBATIONS[name]
    if not args:
        return src
    dst = workdir / f"{src.stem}.{name}.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-an", *args, str(dst)]
    subprocess.run(cmd, check=True)
    return dst


def evaluate(
    analyzer: DeepfakeAnalyzer, rows: pd.DataFrame, video_root: Path, perturbations: list[str]
) -> pd.DataFrame:
    records = []
    with tempfile.TemporaryDirectory(prefix="deepfake-eval-") as tmp:
        for i, row in enumerate(rows.itertuples(index=False), 1):
            src = video_root / str(row.file)
            is_fake = int(str(row.label).strip().upper() == "FAKE")
            for name in perturbations:
                rec = {"file": str(row.file), "perturbation": name, "is_fake": is_fake}
                try:
                    path = perturb(src, name, Path(tmp))
                    t0 = time.perf_counter()
                    result = analyzer.analyze(path)
                    rec.update(
                        p_fake=result.fake_probability,
                        verdict=result.verdict.value,
                        face_coverage=result.face_coverage,
                        latency_ms=(time.perf_counter() - t0) * 1000,
                        status="ok",
                    )
                except NoFaceDetectedError:
                    rec["status"] = "no_face"
                except Exception as exc:  # keep going; failures are reported
                    logger.warning("%s [%s]: %s", src, name, exc)
                    rec["status"] = f"error: {type(exc).__name__}"
                finally:
                    if name != "none":
                        Path(tmp, f"{src.stem}.{name}.mp4").unlink(missing_ok=True)
                records.append(rec)
            logger.info("%d/%d %s", i, len(rows), row.file)
    return pd.DataFrame.from_records(records)


def summarise(df: pd.DataFrame, threshold: float) -> dict:
    out = {}
    for name, part in df.groupby("perturbation", sort=False):
        ok = part[part.status == "ok"]
        report = binary_report(ok.is_fake.tolist(), ok.p_fake.tolist(), threshold) if len(ok) else {}
        report["coverage"] = len(ok) / len(part)
        report["inconclusive_rate"] = float((ok.verdict == "inconclusive").mean()) if len(ok) else 0.0
        report["p50_latency_ms"] = float(ok.latency_ms.median()) if len(ok) else None
        out[name] = report
    return out


def to_markdown(summary: dict) -> str:
    cols = ["auc", "eer", "accuracy", "f1", "ece", "coverage", "p50_latency_ms"]
    lines = ["| perturbation | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, m in summary.items():
        cells = [f"{m[c]:.3f}" if isinstance(m.get(c), float) else str(m.get(c, "-")) for c in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", required=True, help="CSV with file,label (FAKE/REAL)")
    p.add_argument("--video-root", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--perturbations", default="none,h264_crf28,h264_crf35,downscale_0.5,fps_15")
    p.add_argument("--num-clips", type=int, default=3)
    p.add_argument("--face-backend", default="auto")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N videos")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    perturbations = [s.strip() for s in args.perturbations.split(",") if s.strip()]
    unknown = set(perturbations) - PERTURBATIONS.keys()
    if unknown:
        raise SystemExit(f"unknown perturbations: {sorted(unknown)}; choose from {sorted(PERTURBATIONS)}")

    config = DetectorConfig.from_env(
        weights_path=Path(args.weights), num_clips=args.num_clips, face_backend=args.face_backend
    )
    analyzer = DeepfakeAnalyzer.from_config(config)
    rows = pd.read_csv(args.metadata)
    if args.limit:
        rows = rows.head(args.limit)

    df = evaluate(analyzer, rows, Path(args.video_root), perturbations)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "predictions.csv", index=False)
    summary = summarise(df, config.threshold)
    (out / "report.json").write_text(
        json.dumps({"model": analyzer.model_info.model_dump(), "results": summary}, indent=2)
    )
    (out / "report.md").write_text(to_markdown(summary) + "\n")
    print(to_markdown(summary))


if __name__ == "__main__":
    main()
