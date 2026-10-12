# -*- coding: utf-8 -*-
"""Quick end-to-end smoke test with synthetic data (no real dataset needed).

Builds a fake LEVIR-like tree (train/val/test each with A/, B/, label/),
then runs discovery -> datasets -> model -> loss -> metrics -> checkpoint
save/load, to catch pipeline bugs early.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config
import src.data_discovery as dd
from src.dataset import EvalTileDataset, TrainChangeDataset
from src.losses import BCEDiceLoss
from src.metrics import ConfusionMeter
from src.model import SiameseUNet, count_parameters


def make_img(size=512, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    if seed % 3 != 0:  # some samples have change, some not
        mask[100:150, 200:300] = 255
        mask[400:430, 100:180] = 255
    return img, mask


def build_fake(root: Path):
    for split in ("train", "val", "test"):
        n = {"train": 4, "val": 2, "test": 2}[split]
        for sub in ("A", "B", "label"):
            (root / split / sub).mkdir(parents=True)
        for i in range(n):
            seed = hash((split, i)) % 1000
            img, mask = make_img(seed=seed)
            for sub in ("A", "B"):
                from PIL import Image
                Image.fromarray(img + (1 if sub == "B" else 0)).save(
                    root / split / sub / f"{split}_{i}.png")
            from PIL import Image
            Image.fromarray(mask).save(root / split / "label" / f"{split}_{i}.png")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="levir_smoke_"))
    build_fake(tmp)
    print(f"fake dataset at {tmp}")

    dd.OVERRIDES = {}  # force automatic detection
    samples, notes = dd.build_samples(tmp)
    print(f"discovered {len(samples)} samples, notes={notes}")
    assert len(samples) == 8, samples
    assert all(s.split in ("train", "val", "test") for s in samples)

    train_s = [s for s in samples if s.split == "train"]
    val_s = [s for s in samples if s.split == "val"]

    # training dataset path
    ds = TrainChangeDataset(train_s, image_size=256, augment=True)
    dl = DataLoader(ds, batch_size=2, num_workers=0)
    batch = next(iter(dl))
    assert batch["a"].shape == (2, 3, 256, 256), batch["a"].shape
    assert batch["mask"].shape == (2, 1, 256, 256)
    uniq = torch.unique(batch["mask"])
    assert set(uniq.tolist()) <= {0.0, 1.0}, f"mask not binary: {uniq}"
    print("train dataset OK", batch["a"].shape, batch["mask"].shape, uniq)

    # eval tiles path (512 -> 4 tiles per image)
    eds = EvalTileDataset(val_s, image_size=256)
    assert len(eds) == 8, len(eds)
    edl = DataLoader(eds, batch_size=3, num_workers=0)
    ebatch = next(iter(edl))
    m = ebatch["meta"]
    print("eval tiles OK:", ebatch["a"].shape, "valid_hw:", list(zip(
        m["valid_h"].tolist(), m["valid_w"].tolist())),
        "origins:", list(zip(m["top"].tolist(), m["left"].tolist())))

    # model forward/backward
    model = SiameseUNet(base=32)
    print(f"params: {count_parameters(model)/1e6:.2f}M")
    crit = BCEDiceLoss()
    logits = model(batch["a"], batch["b"])
    assert logits.shape == (2, 1, 256, 256)
    loss = crit(logits, batch["mask"])
    loss.backward()
    print("loss:", float(loss))

    # metrics
    meter = ConfusionMeter()
    meter.update(logits, batch["mask"], 0.5)
    print("metrics:", meter.summary())

    # eval forward with tile slicing
    model.eval()
    with torch.no_grad():
        elogits = model(ebatch["a"], ebatch["b"])
    for j in range(elogits.shape[0]):
        assert int(m["valid_h"][j]) == 256 and int(m["valid_w"][j]) == 256
    print("eval forward OK", elogits.shape)

    # checkpoint round trip
    ck = {"model": model.state_dict(), "epoch": 1, "config": {"base_channels": 32}}
    p = tmp / "best.pt"
    torch.save(ck, p)
    m2 = SiameseUNet(base=32)
    m2.load_state_dict(torch.load(p, weights_only=False)["model"])
    print("checkpoint round-trip OK")

    shutil.rmtree(tmp)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    sys.exit(main())
