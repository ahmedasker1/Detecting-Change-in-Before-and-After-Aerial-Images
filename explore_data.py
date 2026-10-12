# -*- coding: utf-8 -*-
"""Step 1 -- inspect the dataset end to end.

Run:  python explore_data.py

What it does:
  1. downloads (or locates) the LEVIR-CD dataset via kagglehub,
  2. prints the directory tree with file counts and sample filenames,
  3. auto-detects A / B / mask folders and matches files by common stem,
  4. verifies sample counts, image sizes, channels and mask values,
  5. reports per-split sample counts and changed/unchanged pixel ratios,
  6. saves a visual sample figure to outputs/figures/data_samples.png
     and a full JSON report to outputs/data_report.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src import config
from src.data_discovery import (build_samples, download_dataset, load_mask,
                                print_tree, split_report, verify_samples)

sys.path.insert(0, str(Path(__file__).resolve().parent))


def save_sample_figure(samples, out_path: Path, n: int = 3) -> None:
    rng = np.random.default_rng(config.SEED)
    picks = rng.choice(len(samples), size=min(n, len(samples)), replace=False)
    fig, axes = plt.subplots(len(picks), 3, figsize=(12, 4 * len(picks)))
    axes = np.atleast_2d(axes)
    for row, idx in enumerate(picks):
        s = samples[int(idx)]
        a = np.array(Image.open(s.a_path).convert("RGB"))
        b = np.array(Image.open(s.b_path).convert("RGB"))
        m = load_mask(s.mask_path)
        axes[row, 0].imshow(a)
        axes[row, 0].set_title(f"{s.stem}  |  t1 (A)")
        axes[row, 1].imshow(b)
        axes[row, 1].set_title("t2 (B)")
        axes[row, 2].imshow(m, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title(f"change mask ({m.mean() * 100:.1f}% changed)")
        for c in range(3):
            axes[row, c].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[explore] Sample figure saved to: {out_path}")


def main() -> None:
    config.ensure_dirs()

    root = download_dataset()
    print_tree(root)

    samples, notes = build_samples(root)
    print(f"\n[explore] Total matched samples: {len(samples)}")
    for i, s in enumerate(samples[:5]):
        print(f"    e.g. {s.stem}: A={s.a_path.name}, B={s.b_path.name}, "
              f"mask={s.mask_path.name} [{s.split}]")

    print("\n[explore] Verification (first/last samples):")
    ver = verify_samples(samples)
    print(f"    image sizes found : {ver['sizes']}")
    print(f"    image modes       : {ver['image_modes']}")
    print(f"    mask unique values: {ver['mask_unique_values']}")
    if ver["problems"]:
        print("    PROBLEMS:")
        for p in ver["problems"]:
            print(f"      - {p}")
    else:
        print("    no problems detected")

    print("\n[explore] Split report:")
    report = split_report(samples)
    for split, st in report.items():
        print(f"    {split:>5}: {st['n_samples']:4d} samples | "
              f"changed pixels {st['changed_frac_mean'] * 100:.2f}% | "
              f"unchanged {st['unchanged_frac_mean'] * 100:.2f}% | "
              f"empty masks {st['empty_masks']}/{st['n_masks']}")

    if notes:
        print("\n[explore] Assumptions / notes:")
        for n in notes:
            print(f"    - {n}")

    # visual samples from the train part
    train_samples = [s for s in samples if s.split == "train"] or samples
    fig_path = config.FIGURES_DIR / "data_samples.png"
    save_sample_figure(train_samples, fig_path, n=3)

    report = {"dataset_root": str(root), "total_samples": len(samples),
              "verification": {"sizes": {str(k): v for k, v in ver["sizes"].items()},
                               "image_modes": ver["image_modes"],
                               "mask_unique_values": ver["mask_unique_values"],
                               "problems": ver["problems"]},
              "splits": report, "notes": notes,
              "samples_head": [f"{s.stem}|{s.split}" for s in samples[:20]]}
    out_json = config.OUTPUT_DIR / "data_report.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[explore] Full report saved to: {out_json}")
    print("[explore] Done.")


if __name__ == "__main__":
    main()
