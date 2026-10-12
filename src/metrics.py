# -*- coding: utf-8 -*-
"""Pixel-level change-detection metrics.

All metrics accumulate a confusion matrix over (TP, FP, FN, TN) pixel counts.

IMPORTANT for imbalanced change detection (~5% changed pixels):
  - The headline metrics are the CHANGE-class Precision/Recall/F1/IoU.
  - Plain "Accuracy" is dominated by the unchanged class and is reported only
    alongside Balanced Accuracy and Cohen's Kappa, which are robust to imbalance.
  - We also report per-class IoU and Mean IoU (mIoU) over the two classes.

Definitions (change class):
    Precision = TP / (TP + FP)      Recall = TP / (TP + FN)
    F1        = 2 P R / (P + R)     IoU    = TP / (TP + FP + FN)
Per-class (unchanged class): TN plays the role of TP, FN<->FP swap.
    Accuracy = (TP + TN) / total          BalancedAcc = 0.5*(Recall_chg + Recall_unch)
    Cohen's Kappa = (p_o - p_e) / (1 - p_e) with observed/expected agreement.

A pixel counts as CHANGE when sigmoid(logit) > threshold (default 0.5).
"""

from __future__ import annotations

import torch


class ConfusionMeter:
    """Accumulates pixel confusion counts across batches.

    NOTE: `update` expects RAW logits -- it applies the sigmoid itself.
    Passing already-sigmoided probabilities double-squashes them
    (sigmoid(sigmoid(x)) > 0.5 for every x), which silently turns every
    prediction into "change" and freezes the reported metrics.
    """

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor,
               threshold: float = 0.5) -> None:
        preds = (torch.sigmoid(logits) > threshold)
        t = targets.bool()
        self.tp += int((preds & t).sum().item())
        self.fp += int((preds & ~t).sum().item())
        self.fn += int((~preds & t).sum().item())
        self.tn += int((~preds & ~t).sum().item())

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    def metrics(self) -> dict:
        eps = 1e-12
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        total = max(self.total, 1)

        # ---- change class (the class of interest) ----
        prec_chg = tp / max(tp + fp, eps)
        rec_chg = tp / max(tp + fn, eps)
        f1_chg = 2 * prec_chg * rec_chg / max(prec_chg + rec_chg, eps)
        iou_chg = tp / max(tp + fp + fn, eps)

        # ---- unchanged class ----
        prec_unch = tn / max(tn + fn, eps)
        rec_unch = tn / max(tn + fp, eps)
        f1_unch = 2 * prec_unch * rec_unch / max(prec_unch + rec_unch, eps)
        iou_unch = tn / max(tn + fp + fn, eps)

        # ---- aggregate / imbalance-robust ----
        acc = (tp + tn) / total
        balanced_acc = 0.5 * (rec_chg + rec_unch)
        p_o = acc
        p_e = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total * total)
        kappa = (p_o - p_e) / max(1.0 - p_e, eps)
        miou = 0.5 * (iou_chg + iou_unch)

        return {
            # change class (headline)
            "precision": prec_chg, "recall": rec_chg, "f1": f1_chg, "iou": iou_chg,
            # unchanged class
            "precision_unchanged": prec_unch, "recall_unchanged": rec_unch,
            "f1_unchanged": f1_unch, "iou_unchanged": iou_unch,
            # aggregate
            "mean_iou": miou, "accuracy": acc, "balanced_accuracy": balanced_acc,
            "kappa": kappa,
            "changed_pixel_frac": (tp + fn) / total,
            "pred_changed_pixel_frac": (tp + fp) / total,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    def summary(self) -> str:
        m = self.metrics()
        return (f"change: P={m['precision']:.4f} R={m['recall']:.4f} "
                f"F1={m['f1']:.4f} IoU={m['iou']:.4f} | mIoU={m['mean_iou']:.4f} "
                f"| balAcc={m['balanced_accuracy']:.4f} kappa={m['kappa']:.4f}")
