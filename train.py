# -*- coding: utf-8 -*-
"""Step 2 -- train the Siamese U-Net.

Run:  python train.py [--epochs 60 --batch-size 16 --lr 1e-3 --image-size 256]

What it does:
  1. discovers/matches the dataset (same logic as explore_data.py),
  2. builds train / val loaders with synchronised augmentations,
  3. trains with BCE + Dice loss, logging loss/P/R/F1/IoU every epoch,
  4. early-stops on validation IoU and always keeps the best checkpoint,
  5. writes everything into a NEW timestamped run folder under outputs/
     (previous runs are never overwritten).

All artifacts of a run live in:  outputs/run_YYYYmmdd_HHMMSS/
  config.json, history.csv, best_model.pt, last_model.pt, curves.png, train_log.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config
from src.data_discovery import build_samples, change_ratio_stats, download_dataset
from src.dataset import EvalTileDataset, TrainChangeDataset, samples_for_split
from src.losses import BCEDiceLoss
from src.metrics import ConfusionMeter
from src.model import SiameseUNet, count_parameters


# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Siamese U-Net on LEVIR-CD")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    p.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    p.add_argument("--base-channels", type=int, default=config.BASE_CHANNELS)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE)
    p.add_argument("--amp", type=int, default=1, help="1 = mixed precision (GPU), 0 = off")
    p.add_argument("--resume", type=str, default=None,
                   help="path to a checkpoint (best_model.pt) to continue training from "
                        "(loads model weights only; fresh optimizer/scheduler)")
    return p.parse_args()


def make_run_dir() -> Path:
    """Timestamped run folder -> never overwrites a previous run."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = config.OUTPUT_DIR / f"run_{stamp}"
    n = 1
    while run_dir.exists():  # extra safety within the same second
        run_dir = config.OUTPUT_DIR / f"run_{stamp}_{n}"
        n += 1
    run_dir.mkdir(parents=True)
    return run_dir


class Logger:
    """Tee stdout into a log file (line-buffered so progress is visible)."""

    def __init__(self, path: Path):
        self.file = open(path, "w", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout

    def write(self, msg):
        self.stdout.write(msg)
        self.stdout.flush()
        self.file.write(msg)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, scaler,
              threshold, amp, grad_clip) -> tuple[float, dict]:
    model.train()
    meter = ConfusionMeter()
    losses = []
    for batch in loader:
        a = batch["a"].to(device, non_blocking=True)
        b = batch["b"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(a, b)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(a, b)
            loss = criterion(logits, y)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        losses.append(loss.item())
        meter.update(logits.float(), y, threshold)
    return float(np.mean(losses)), meter.metrics()


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device, threshold, amp) -> tuple[float, dict]:
    """Validation over deterministic tiles; reflect-padding is excluded."""
    model.eval()
    meter = ConfusionMeter()
    losses = []
    for batch in loader:
        a = batch["a"].to(device, non_blocking=True)
        b = batch["b"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        if amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(a, b)
                loss = criterion(logits, y)
        else:
            logits = model(a, b)
            loss = criterion(logits, y)
        losses.append(loss.item())
        # ConfusionMeter applies sigmoid internally -> pass RAW logits.
        metas = batch["meta"]
        for j in range(logits.shape[0]):
            vh = int(metas["valid_h"][j])
            vw = int(metas["valid_w"][j])
            meter.update(logits.float()[j, :, :vh, :vw],
                         y[j, :, :vh, :vw], threshold)
    return float(np.mean(losses)), meter.metrics()


def plot_curves(history: list[dict], out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    ax[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    ax[0].set_title("Loss (BCE + Dice)")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    for key, title in (("iou", "IoU"), ("f1", "F1")):
        ax[1].plot(epochs, [h[f"train_{key}"] for h in history], label=f"train {title}")
        ax[1].plot(epochs, [h[f"val_{key}"] for h in history], label=f"val {title}")
        ax[1].set_title(title)
        ax[1].set_xlabel("epoch")
        ax[1].legend()
        ax[1].grid(alpha=0.3)
    ax[2].plot(epochs, [h["lr"] for h in history])
    ax[2].set_title("Learning rate")
    ax[2].set_xlabel("epoch")
    ax[2].set_yscale("log")
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    config.ensure_dirs()
    run_dir = make_run_dir()
    sys.stdout = Logger(run_dir / "train_log.txt")

    print("=" * 70)
    print("Siamese U-Net change detection -- training")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.amp) and device.type == "cuda"
    print(f"Device          : {device}" + (f" ({torch.cuda.get_device_name(0)})"
                                            if device.type == "cuda" else " (CPU)"))
    print(f"Mixed precision : {amp}")
    print(f"Run folder      : {run_dir}")

    # ---- data ----
    root = download_dataset()
    samples, notes = build_samples(root)
    train_s = samples_for_split(samples, "train")
    val_s = samples_for_split(samples, "val")
    if not train_s or not val_s:
        sys.exit("ERROR: empty train or validation split -- run explore_data.py first.")
    print(f"Samples         : train={len(train_s)}, val={len(val_s)}")
    for n in notes:
        print(f"  note: {n}")

    train_ds = TrainChangeDataset(train_s, image_size=args.image_size, augment=True)
    val_ds = EvalTileDataset(val_s, image_size=args.image_size)  # deterministic tiles
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          persistent_workers=args.num_workers > 0,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False)

    # ---- model / loss / optim ----
    model = SiameseUNet(in_ch=config.IN_CHANNELS,
                        base=args.base_channels).to(device)
    print(f"Model           : SiameseUNet (base={args.base_channels}), "
          f"{count_parameters(model) / 1e6:.2f} M parameters")
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"Resumed weights : {args.resume} (checkpoint epoch {ck.get('epoch')}, "
              f"val IoU {ck.get('val_iou', float('nan')):.4f}) -- continuing training")

    # auto pos_weight from the train split: neg/pos pixel ratio (capped)
    pos_weight = None
    if config.USE_AUTO_POS_WEIGHT:
        frac = change_ratio_stats(train_s, max_n=config.MAX_MASKS_FOR_STATS)["changed_frac_mean"]
        if 0.0 < frac < 0.5:
            pos_weight = min((1.0 - frac) / frac, config.MAX_POS_WEIGHT)
            print(f"Class imbalance : {frac * 100:.2f}% changed pixels -> "
                  f"BCE pos_weight = {pos_weight:.2f}")

    criterion = BCEDiceLoss(config.BCE_WEIGHT, config.DICE_WEIGHT,
                            pos_weight=pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE)
    scaler = torch.amp.GradScaler(enabled=amp)

    # ---- save run config ----
    run_cfg = config.as_dict()
    run_cfg.update(vars(args))
    run_cfg["device"] = str(device)
    run_cfg["n_train"] = len(train_s)
    run_cfg["n_val"] = len(val_s)
    run_cfg["criterion"] = ("BCEWithLogits + Dice (0.5/0.5)"
                            + (f" with pos_weight={pos_weight:.2f}" if pos_weight else ""))
    if args.resume:
        run_cfg["resume_from"] = str(args.resume)
    (run_dir / "config.json").write_text(json.dumps(run_cfg, indent=2),
                                         encoding="utf-8")

    # ---- training loop ----
    history: list[dict] = []
    best_iou = -1.0
    best_epoch = -1
    bad_epochs = 0
    threshold = config.THRESHOLD

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_m = run_epoch(model, train_dl, criterion, optimizer,
                                  device, scaler, threshold, amp, config.GRAD_CLIP)
        va_loss, va_m = evaluate_epoch(model, val_dl, criterion, device,
                                       threshold, amp)
        scheduler.step(va_m["iou"])
        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        rec = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
               "train_precision": tr_m["precision"], "train_recall": tr_m["recall"],
               "train_f1": tr_m["f1"], "train_iou": tr_m["iou"],
               "train_accuracy": tr_m["accuracy"],
               "val_precision": va_m["precision"], "val_recall": va_m["recall"],
               "val_f1": va_m["f1"], "val_iou": va_m["iou"],
               "val_accuracy": va_m["accuracy"], "lr": lr_now, "seconds": dt}
        history.append(rec)

        improved = va_m["iou"] > best_iou
        if improved:
            best_iou = va_m["iou"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_iou": va_m["iou"], "val_f1": va_m["f1"],
                        "config": run_cfg},
                       run_dir / "best_model.pt")
        else:
            bad_epochs += 1
        torch.save({"model": model.state_dict(), "epoch": epoch,
                    "config": run_cfg}, run_dir / "last_model.pt")

        with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)
        plot_curves(history, run_dir / "curves.png")

        star = "  *best*" if improved else ""
        print(f"epoch {epoch:3d} | train loss {tr_loss:.4f} "
              f"P {tr_m['precision']:.3f} R {tr_m['recall']:.3f} "
              f"F1 {tr_m['f1']:.3f} IoU {tr_m['iou']:.3f}")
        print(f"          | val   loss {va_loss:.4f} "
              f"P {va_m['precision']:.3f} R {va_m['recall']:.3f} "
              f"F1 {va_m['f1']:.3f} IoU {va_m['iou']:.3f} "
              f"({dt:.1f}s, lr={lr_now:.2e}){star}")

        if bad_epochs >= args.patience:
            print(f"Early stopping: no val IoU improvement for {args.patience} epochs.")
            break

    print("-" * 70)
    print(f"Best epoch: {best_epoch}  |  best val IoU: {best_iou:.4f}")
    print(f"Best checkpoint : {run_dir / 'best_model.pt'}")
    print(f"History         : {run_dir / 'history.csv'}")
    print(f"Curves          : {run_dir / 'curves.png'}")
    print(f"Next step       : python evaluate.py --run-dir {run_dir}")


if __name__ == "__main__":
    main()
