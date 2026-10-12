"""Train or fine-tune the detector.

    python -m training.train --metadata data/metadata.csv --video-root data/faces \
        --output-dir runs/exp1 --temporal-mode sequence --epochs 20

Fixes over the original ``train.py``: device-agnostic (no hard-coded
``torch.cuda.LongTensor``), group-aware splits, held-out test set, AUC computed from
probabilities, augmentation, AMP, gradient clipping, model selection on validation AUC,
a non-shuffled validation loader, and no interactive ``input()`` prompt.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deepfake_detector.config import FAKE_INDEX
from deepfake_detector.metrics import binary_report
from deepfake_detector.model import DeepfakeDetector, ModelSpec, load_checkpoint, save_checkpoint
from training.data import ClipAugment, VideoClipDataset, load_samples, split_samples

logger = logging.getLogger("train")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", required=True, help="CSV with file,label[,group]")
    p.add_argument("--video-root", required=True, help="directory containing face-cropped videos")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--init-from", help="checkpoint to fine-tune (v2 or the original state_dict)")
    p.add_argument("--temporal-mode", choices=["legacy", "sequence"], default="sequence")
    p.add_argument("--sequence-length", type=int, default=80)
    p.add_argument("--image-size", type=int, default=112)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr-scale", type=float, default=0.1, help="LR multiplier for the CNN")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--freeze-backbone-epochs", type=int, default=1)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-pretrained", action="store_true", help="random-init backbone (tests/offline)")
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, amp=False) -> dict:
    """One pass over ``loader``; trains when ``optimizer`` is given. Returns loss + metrics."""

    training = optimizer is not None
    model.train(training)
    total_loss, n = 0.0, 0
    y_fake, p_fake = [], []
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            _, logits = model(x)
            loss = criterion(logits, y)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * len(y)
        n += len(y)
        y_fake += (y == FAKE_INDEX).long().tolist()
        p_fake += torch.softmax(logits.float(), dim=-1)[:, FAKE_INDEX].detach().cpu().tolist()
        if training and step % 20 == 0:
            logger.info("  step %d/%d loss %.4f", step, len(loader), total_loss / n)
    report = binary_report(y_fake, p_fake) if n else {}
    report["loss"] = total_loss / max(n, 1)
    return report


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed_everything(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)

    samples = load_samples(args.metadata, args.video_root)
    train_s, val_s, test_s = split_samples(samples, args.val_ratio, args.test_ratio, args.seed)
    for name, part in (("train", train_s), ("val", val_s), ("test", test_s)):
        fakes = sum(s.label == FAKE_INDEX for s in part)
        logger.info("%s: %d videos (%d fake, %d real)", name, len(part), fakes, len(part) - fakes)
        (out / f"split_{name}.txt").write_text("\n".join(str(s.path) for s in part))

    common = dict(sequence_length=args.sequence_length, image_size=args.image_size)
    train_ds = VideoClipDataset(
        train_s, augment=None if args.no_augment else ClipAugment(), random_start=True, **common
    )
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_dl = DataLoader(VideoClipDataset(val_s, **common), shuffle=False, **loader_kw)
    test_dl = DataLoader(VideoClipDataset(test_s, **common), shuffle=False, **loader_kw)

    if args.init_from:
        model, _ = load_checkpoint(args.init_from, device="cpu", temporal_mode=args.temporal_mode)
    else:
        model = DeepfakeDetector(
            ModelSpec(temporal_mode=args.temporal_mode, lstm_bias=True), pretrained_backbone=not args.no_pretrained
        )
    model.to(device)

    # Class-balanced loss: deepfake datasets are typically fake-heavy.
    counts = np.bincount([s.label for s in train_s], minlength=2).astype(float)
    weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    backbone = list(model.model.parameters())
    head = [p for n, p in model.named_parameters() if not n.startswith("model.")]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": args.lr * args.backbone_lr_scale}, {"params": head, "lr": args.lr}],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")

    history, best_auc, best_epoch, stale = [], -1.0, 0, 0
    best_path = out / "best.pt"
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_backbone_epochs
        for p in backbone:
            p.requires_grad_(not frozen)
        logger.info("epoch %d/%d%s", epoch, args.epochs, " (backbone frozen)" if frozen else "")
        train_m = run_epoch(model, train_dl, device, criterion, optimizer, scaler, args.amp)
        val_m = run_epoch(model, val_dl, device, criterion, amp=args.amp)
        scheduler.step()
        history.append({"epoch": epoch, "train": train_m, "val": val_m})
        (out / "history.json").write_text(json.dumps(history, indent=2))
        val_auc = val_m.get("auc", -val_m["loss"])
        logger.info(
            "epoch %d: train loss %.4f auc %.4f | val loss %.4f auc %.4f",
            epoch,
            train_m["loss"],
            train_m.get("auc", float("nan")),
            val_m["loss"],
            val_m.get("auc", float("nan")),
        )
        if val_auc > best_auc:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            save_checkpoint(model, best_path, epoch=epoch, val=val_m, args=vars(args))
        else:
            stale += 1
            if stale >= args.patience:
                logger.info("early stopping: no val AUC improvement for %d epochs", args.patience)
                break

    model, _ = load_checkpoint(best_path, device=device)
    test_m = run_epoch(model, test_dl, device, criterion, amp=args.amp)
    summary = {
        "best_epoch": best_epoch,
        "val_auc": best_auc,
        "test": test_m,
        "spec": asdict(model.spec),
        "minutes": round((time.time() - start) / 60, 2),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("test: %s", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in test_m.items()}))


if __name__ == "__main__":
    main()
