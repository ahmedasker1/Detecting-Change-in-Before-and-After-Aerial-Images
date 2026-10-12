# -*- coding: utf-8 -*-
"""Diagnose the frozen-val-metrics problem: compare model.eval() vs model.train()
(batch statistics) on the SAME validation tiles, on CPU (training is on GPU)."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config
from src.data_discovery import build_samples, download_dataset
from src.dataset import EvalTileDataset, samples_for_split
from src.metrics import ConfusionMeter
from src.model import SiameseUNet

ckpt_path = Path(r"D:\DEPI Project\outputs\run_20261009_193121\last_model.pt")
device = torch.device("cpu")

root = download_dataset()
samples, _ = build_samples(root)
val_s = samples_for_split(samples, "val")
ds = EvalTileDataset(val_s, image_size=256)
dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

model = SiameseUNet(base=32)
model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model"])

batch = next(iter(dl))
a, b, y = batch["a"], batch["b"], batch["mask"]
print("input a mean/std:", float(a.mean()), float(a.std()))
print("mask changed frac in batch:", float(y.mean()))

for mode in ("eval", "train"):
    if mode == "eval":
        model.eval()
    else:
        model.train()
    with torch.no_grad():
        logits = model(a, b)
    probs = torch.sigmoid(logits)
    meter = ConfusionMeter()
    meter.update(logits, y, 0.5)
    m = meter.metrics()
    qs = [float(torch.quantile(probs, q)) for q in (0.01, 0.25, 0.5, 0.75, 0.99)]
    print(f"[{mode:5}] prob quantiles 1/25/50/75/99%: "
          + " ".join(f"{q:.3f}" for q in qs)
          + f" | IoU={m['iou']:.3f} P={m['precision']:.3f} R={m['recall']:.3f}")
    # best threshold on these 8 tiles (diagnostic only)
    best = (0.0, 0.0)
    for thr in np.arange(0.3, 0.95, 0.05):
        mm = ConfusionMeter()
        mm.update(logits, y, float(thr))
        if mm.metrics()["iou"] > best[1]:
            best = (float(thr), mm.metrics()["iou"])
    print(f"[{mode:5}] best threshold on this batch: {best[0]:.2f} -> IoU {best[1]:.3f}")
