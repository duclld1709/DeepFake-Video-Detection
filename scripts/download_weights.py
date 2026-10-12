"""Download the published ``80_frame`` checkpoint into ``weights/``.

python scripts/download_weights.py                 # Google Drive (default)
python scripts/download_weights.py --sha256 <hex>  # verify integrity
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

GDRIVE_ID = "1JSMl_gM1ZYoZgAH45U8PzJsu-qXh2duO"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default="weights/80_frame")
    p.add_argument("--sha256", help="expected checksum")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"{out} already exists (use --force to re-download)")
    else:
        try:
            import gdown
        except ImportError:
            print("pip install gdown", file=sys.stderr)
            return 1
        out.parent.mkdir(parents=True, exist_ok=True)
        if gdown.download(id=GDRIVE_ID, output=str(out), quiet=False) is None:
            print("download failed", file=sys.stderr)
            return 1

    digest = sha256(out)
    print(f"sha256 {digest}")
    if args.sha256 and digest != args.sha256.lower():
        print("checksum mismatch", file=sys.stderr)
        return 1

    from deepfake_detector.model import load_checkpoint

    model, meta = load_checkpoint(out)
    print(f"loaded OK: temporal_mode={model.spec.temporal_mode}, lstm_bias={model.spec.lstm_bias}, {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
