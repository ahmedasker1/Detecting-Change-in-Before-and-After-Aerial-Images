# -*- coding: utf-8 -*-
"""Step 3 -- evaluate the best checkpoint on the test set.

Run:  python evaluate.py [--run-dir outputs/run_YYYYmmdd_HHMMSS] [--threshold 0.5|auto]

Loads `<run-dir>/best_model.pt`, evaluates ONCE on the test split, and saves
into `<run-dir>/eval_<stamp>/`:

  - test_metrics.json   (all metrics + confusion matrix + threshold sweep)
  - confusion_matrix.png (raw counts + row-normalized %)
  - predictions/        (qualitative figures: A, B, GT mask, predicted mask)

Metrics are dominated by the CHANGE class (the class of interest). Because
~95% of pixels are unchanged, plain Accuracy is reported only next to
Balanced Accuracy and Cohen's Kappa, plus per-class Precision/Recall/F1/IoU
and Mean IoU -- see src/metrics.py for the exact formulas.

`--threshold auto` sweeps thresholds on the VALIDATION split first and uses
the one with the best validation F1 on the test set (printed and stored).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from src import config
from src.data_discovery import build_samples, download_dataset, load_mask
from src.dataset import EvalTileDataset, samples_for_split
from src.metrics import ConfusionMeter
from src.model import SiameseUNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Siamese U-Net on LEVIR-CD test")
    p.add_argument("--run-dir", type=str, default=None,
                   help="outputs/run_... folder (default: the newest run)")
    p.add_argument("--threshold", type=str, default="0.5",
                   help="'0.5' or 'auto' = sweep thresholds on the validation split")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE * 2)
    p.add_argument("--num-vis", type=int, default=config.NUM_VIS_SAMPLES)
    p.add_argument("--checkpoint", type=str, default="best_model.pt")
    return p.parse_args()


def newest_run_dir() -> Path:
    """Best run by default: highest val_iou among runs that have a checkpoint.

    (Picking simply the *newest* folder would grab throwaway demo runs such as
    the 5-epoch notebook training in run_nb_*/.)"""
    import torch as _torch

    runs = sorted(config.OUTPUT_DIR.glob("run_*"))
    if not runs:
        sys.exit("ERROR: no run folder under outputs/. Train first:  python train.py")
    best_run, best_val = None, -1.0
    for r in runs:
        ck = r / "best_model.pt"
        if not ck.exists():
            continue
        meta = _torch.load(ck, map_location="cpu", weights_only=False)
        v = float(meta.get("val_iou", -1.0))
        print(f"  candidate {r.name}: epoch {meta.get('epoch')} val_iou {v:.4f}")
        if v > best_val:
            best_run, best_val = r, v
    if best_run is None:
        sys.exit("ERROR: no run contains best_model.pt.")
    print(f"  -> selected {best_run.name} (highest val IoU)")
    return best_run


# ---------------------------------------------------------------------------
@torch.no_grad()
def _predict_logits(model, loader, device):
    """One pass over tiles -> yields (logits[cpu], mask[cpu], meta batch)."""
    for batch in loader:
        a = batch["a"].to(device, non_blocking=True)
        b = batch["b"].to(device, non_blocking=True)
        logits = model(a, b).float().cpu()
        yield logits, batch["mask"], batch["meta"]


@torch.no_grad()
def sweep_threshold_on_val(model, val_samples, device, batch_size) -> tuple[float, list[dict]]:
    """Try thresholds 0.30..0.70 on validation tiles, pick best F1."""
    ds = EvalTileDataset(val_samples, image_size=config.IMAGE_SIZE)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=config.NUM_WORKERS, pin_memory=True)
    thresholds = [round(float(t), 2) for t in np.arange(0.30, 0.701, 0.05)]
    meters = {t: ConfusionMeter() for t in thresholds}
    for logits, masks, metas in _predict_logits(model, dl, device):
        for j in range(logits.shape[0]):
            vh, vw = int(metas["valid_h"][j]), int(metas["valid_w"][j])
            lg, yt = logits[j, :, :vh, :vw], masks[j, :, :vh, :vw]
            for t in thresholds:
                meters[t].update(lg.unsqueeze(0), yt.unsqueeze(0), t)
    rows = []
    for t in thresholds:
        m = meters[t].metrics()
        rows.append({"threshold": t, "f1": m["f1"], "iou": m["iou"],
                     "precision": m["precision"], "recall": m["recall"]})
        print(f"    thr {t:.2f}: val F1={m['f1']:.4f}  IoU={m['iou']:.4f}  "
              f"P={m['precision']:.4f}  R={m['recall']:.4f}")
    best = max(rows, key=lambda r: r["f1"])
    print(f"  -> chosen threshold {best['threshold']:.2f} "
          f"(best validation F1 = {best['f1']:.4f})")
    return best["threshold"], rows


# ---------------------------------------------------------------------------
@torch.no_grad()
def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else newest_run_dir()
    ckpt_path = run_dir / args.checkpoint
    if not ckpt_path.exists():
        sys.exit(f"ERROR: checkpoint not found: {ckpt_path}")
    eval_dir = run_dir / f"eval_{time.strftime('%Y%m%d_%H%M%S')}"
    eval_dir.mkdir(parents=True)
    (eval_dir / "predictions").mkdir()
    print(f"Run folder      : {run_dir}")
    print(f"Evaluation dir  : {eval_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device          : {device}")

    # ---- data (same discovery/split logic as training) ----
    root = download_dataset()
    samples, _ = build_samples(root)
    test_s = samples_for_split(samples, "test")
    val_s = samples_for_split(samples, "val")
    if not test_s:
        sys.exit("ERROR: empty test split.")
    print(f"Test samples    : {len(test_s)}")

    ds = EvalTileDataset(test_s, image_size=config.IMAGE_SIZE)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=config.NUM_WORKERS, pin_memory=True)

    # ---- model ----
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    base = int(cfg.get("base-channels", cfg.get("base_channels", config.BASE_CHANNELS)))
    model = SiameseUNet(in_ch=config.IN_CHANNELS, base=base).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Checkpoint      : {ckpt_path} (epoch {ckpt.get('epoch', '?')}, "
          f"val IoU {ckpt.get('val_iou', float('nan')):.4f})")

    # ---- threshold (fixed or tuned on validation) ----
    sweep_rows = None
    if args.threshold.lower() == "auto":
        print(f"\nThreshold sweep on VALIDATION ({len(val_s)} images):")
        threshold, sweep_rows = sweep_threshold_on_val(model, val_s, device,
                                                       args.batch_size)
    else:
        threshold = float(args.threshold)
    print(f"Threshold for 'change' pixel: sigmoid(logits) > {threshold}")

    # ---- evaluate tile by tile; metrics exclude any reflect-padding ----
    meter = ConfusionMeter()
    n_tiles = 0
    t0 = time.time()
    for logits, masks, metas in _predict_logits(model, dl, device):
        for j in range(logits.shape[0]):
            vh, vw = int(metas["valid_h"][j]), int(metas["valid_w"][j])
            meter.update(logits[j, :, :vh, :vw],
                         masks[j, :, :vh, :vw], threshold)
        n_tiles += logits.shape[0]
    dt = time.time() - t0

    m = meter.metrics()
    changed_pct = m["changed_pixel_frac"] * 100
    print("-" * 70)
    print(f"Evaluated {n_tiles} tiles of {len(test_s)} test images "
          f"({config.IMAGE_SIZE}x{config.IMAGE_SIZE}px each) in {dt:.1f}s")
    print("\n== CHANGE-class metrics (headline) ==")
    print(f"Precision : {m['precision']:.4f}")
    print(f"Recall    : {m['recall']:.4f}")
    print(f"F1-score  : {m['f1']:.4f}")
    print(f"IoU       : {m['iou']:.4f}")
    print("\n== Aggregate (robust to imbalance) ==")
    print(f"Mean IoU            : {m['mean_iou']:.4f}  "
          f"(change {m['iou']:.4f} / unchanged {m['iou_unchanged']:.4f})")
    print(f"Balanced Accuracy   : {m['balanced_accuracy']:.4f}")
    print(f"Cohen's Kappa       : {m['kappa']:.4f}")
    print(f"Overall Accuracy    : {m['accuracy']:.4f}  <-- NOT informative here: "
          f"{100 - changed_pct:.2f}% of pixels are unchanged")
    print("\n== Per-class table ==")
    print(f"{'class':>10} | {'precision':>9} | {'recall':>9} | {'f1':>9} | {'iou':>9}")
    print(f"{'change':>10} | {m['precision']:9.4f} | {m['recall']:9.4f} | "
          f"{m['f1']:9.4f} | {m['iou']:9.4f}")
    print(f"{'unchanged':>10} | {m['precision_unchanged']:9.4f} | "
          f"{m['recall_unchanged']:9.4f} | {m['f1_unchanged']:9.4f} | "
          f"{m['iou_unchanged']:9.4f}")
    print(f"\nConfusion (pixels): TP={m['tp']:,}  FP={m['fp']:,}  "
          f"FN={m['fn']:,}  TN={m['tn']:,}")

    # ---- save metrics ----
    out = {"checkpoint": str(ckpt_path), "threshold": threshold,
           "threshold_mode": "auto(val-F1)" if sweep_rows else "fixed",
           "threshold_sweep_val": sweep_rows,
           "n_test_images": len(test_s), "n_tiles": n_tiles,
           "tile_size": config.IMAGE_SIZE,
           "metric_definition": (
               "Pixel-level over the full test set. Change class: "
               "Precision=TP/(TP+FP), Recall=TP/(TP+FN), F1=2PR/(P+R), "
               "IoU=TP/(TP+FP+FN); unchanged class uses TN as TP. "
               "MeanIoU=mean of both classes' IoU. BalancedAcc=mean of the two "
               "recalls. Kappa=(p_o-p_e)/(1-p_e). Pred = sigmoid(logits) > threshold."),
           **m}
    (eval_dir / "test_metrics.json").write_text(json.dumps(out, indent=2),
                                                encoding="utf-8")

    # ---- confusion matrix figure: counts+% and row-normalized ----
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=float)
    cm_pct = cm / cm.sum() * 100.0
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_row = cm / cm.sum(axis=1, keepdims=True) * 100.0
    labels = [["TN", "FP"], ["FN", "TP"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, mat, fmt, title in (
            (axes[0], cm_pct, lambda v: f"{v:.2f}%",
             "% of all pixels"),
            (axes[1], cm_row, lambda v: f"{v:.1f}%",
             "% of the TRUE class (row-normalized)")):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                txt = f"{labels[i][j]}\n{cm[i, j]:,.0f}\n{fmt(mat[i, j])}"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if mat[i, j] > 50 else "black", fontsize=9)
        ax.set_xticks([0, 1], ["pred: unchanged", "pred: change"])
        ax.set_yticks([0, 1], ["true: unchanged", "true: change"])
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"Pixel confusion matrix (test, threshold={threshold})")
    fig.tight_layout()
    fig.savefig(eval_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)
    print(f"Confusion matrix: {eval_dir / 'confusion_matrix.png'}")

    # ---- qualitative figures: reassemble full-size predictions ----
    vis_ids = list(range(min(args.num_vis, len(test_s))))
    preds: dict[int, np.ndarray] = {}
    done: set[tuple[int, int, int]] = set()
    for logits, masks, metas in _predict_logits(model, dl, device):
        for j in range(logits.shape[0]):
            sid = int(metas["sample_idx"][j])
            if sid not in vis_ids:
                continue
            pred = (torch.sigmoid(logits[j, 0]) > threshold).numpy().astype(np.uint8)
            vh, vw = int(metas["valid_h"][j]), int(metas["valid_w"][j])
            top, left = int(metas["top"][j]), int(metas["left"][j])
            if sid not in preds:
                w, h = Image.open(test_s[sid].a_path).size
                preds[sid] = np.zeros((h, w), dtype=np.uint8)
            preds[sid][top:top + vh, left:left + vw] = pred[:vh, :vw]
            done.add((sid, top, left))
        if all(all((sid, t, l) in done
                   for t, l in _tile_grid(test_s[sid], config.IMAGE_SIZE))
               for sid in vis_ids):
            break

    for sid in vis_ids:
        s = test_s[sid]
        a = np.array(Image.open(s.a_path).convert("RGB"))
        b = np.array(Image.open(s.b_path).convert("RGB"))
        gt = load_mask(s.mask_path)
        fig, ax = plt.subplots(1, 4, figsize=(18, 5))
        ax[0].imshow(a); ax[0].set_title("t1 (A)")
        ax[1].imshow(b); ax[1].set_title("t2 (B)")
        ax[2].imshow(gt, cmap="gray", vmin=0, vmax=1)
        ax[2].set_title(f"ground truth ({gt.mean() * 100:.1f}% change)")
        ax[3].imshow(preds[sid], cmap="gray", vmin=0, vmax=1)
        ax[3].set_title(f"prediction (thr={threshold})")
        for k in range(4):
            ax[k].axis("off")
        fig.suptitle(f"test sample: {s.stem}")
        fig.tight_layout()
        fp = eval_dir / "predictions" / f"pred_{s.stem}.png"
        fig.savefig(fp, dpi=110)
        plt.close(fig)
    print(f"Qualitative figs: {eval_dir / 'predictions'} ({len(vis_ids)} samples)")
    print(f"All metrics JSON: {eval_dir / 'test_metrics.json'}")
    print("Done.")


def _tile_grid(sample, size: int):
    w, h = Image.open(sample.a_path).size
    return [(t, l) for t in range(0, h, size) for l in range(0, w, size)]


if __name__ == "__main__":
    main()
