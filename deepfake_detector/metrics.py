"""Binary detection metrics. ``y_true``: 1 = fake, ``scores``: P(fake)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def equal_error_rate(y_true: Sequence[int], scores: Sequence[float]) -> tuple[float, float]:
    """Return ``(eer, threshold)`` where false-positive and false-negative rates meet."""

    fpr, tpr, thresholds = roc_curve(y_true, scores, drop_intermediate=False)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thresholds[i])


def expected_calibration_error(y_true: Sequence[int], scores: Sequence[float], n_bins: int = 10) -> float:
    """ECE over equal-width bins of the predicted-class confidence."""

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs((pred[in_bin] == y[in_bin]).mean() - conf[in_bin].mean())
    return float(ece)


def binary_report(y_true: Sequence[int], scores: Sequence[float], threshold: float = 0.5) -> dict[str, float]:
    """Threshold-free (AUC, EER, ECE) and thresholded metrics in one dict."""

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    report = {
        "n": int(len(y)),
        "n_fake": int(y.sum()),
        "accuracy": float((pred == y).mean()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold": float(threshold),
        "ece": expected_calibration_error(y, p),
    }
    if len(np.unique(y)) == 2:
        # The original script passed hard labels to roc_auc_score; AUC needs scores.
        report["auc"] = float(roc_auc_score(y, p))
        report["eer"], report["eer_threshold"] = equal_error_rate(y, p)
    return report
