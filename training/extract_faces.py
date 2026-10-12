"""Turn raw videos into face-cropped training videos with the *same* detector and
tracker the inference pipeline uses (no train/serve skew).

    python -m training.extract_faces --input-dir data/raw --output-dir data/faces \
        --max-frames 150 --size 224
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2

from deepfake_detector.faces import FaceTracker, build_face_detector
from deepfake_detector.preprocessing import crop
from deepfake_detector.video import iter_frames, probe

logger = logging.getLogger("extract_faces")
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def extract(src: Path, dst: Path, tracker: FaceTracker, max_frames: int, size: int) -> int:
    info = probe(src)
    tracker.reset()
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (size, size))
    written = 0
    try:
        for _, frame in iter_frames(src):
            box = tracker.update(frame)
            if box is None:
                continue
            face = crop(frame, box)
            if face.size == 0:
                continue
            writer.write(cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA))
            written += 1
            if written >= max_frames:
                break
    finally:
        writer.release()
    if written == 0:
        dst.unlink(missing_ok=True)
    return written


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-frames", type=int, default=150)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--face-backend", default="auto")
    p.add_argument("--detect-every", type=int, default=1)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tracker = FaceTracker(build_face_detector(args.face_backend), detect_every=args.detect_every)
    src_root, dst_root = Path(args.input_dir), Path(args.output_dir)
    videos = sorted(v for v in src_root.rglob("*") if v.suffix.lower() in VIDEO_SUFFIXES)
    for i, src in enumerate(videos, 1):
        dst = dst_root / src.relative_to(src_root).with_suffix(".mp4")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        n = extract(src, dst, tracker, args.max_frames, args.size)
        logger.info("%d/%d %s -> %d face frames", i, len(videos), src.name, n)


if __name__ == "__main__":
    main()
