# -*- coding: utf-8 -*-
"""Automatic discovery of the LEVIR-CD dataset layout.

This module never assumes folder names in advance. It:
  1. downloads the dataset with kagglehub (cached across runs),
  2. walks the directory tree and prints it,
  3. classifies every image folder as "A" (time 1), "B" (time 2) or
     "mask" (change label) using conservative keyword rules,
  4. groups A/B/mask folders that share the same parent and matches
     files by common file stem (identifier),
  5. detects official train/val/test splits if the dataset ships them,
     otherwise falls back to a seeded 70/15/15 random split,
  6. verifies every sample (three files present, sizes match, masks binary).

All folder heuristics can be overridden by editing `OVERRIDES` below.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src import config

# ---------------------------------------------------------------------------
# Keyword rules (case-insensitive). Single letters are matched EXACTLY to
# avoid false positives; longer keywords are matched as substrings.
# ---------------------------------------------------------------------------
A_KEYS_EXACT = {"a", "t1", "im1"}
A_KEYS_SUBSTR = {"image1", "images1", "img1", "imgs1", "time1", "before", "im1"}
B_KEYS_EXACT = {"b", "t2", "im2"}
B_KEYS_SUBSTR = {"image2", "images2", "img2", "imgs2", "time2", "after", "im2"}
MASK_KEYS_EXACT = {"mask", "masks", "label", "labels", "gt", "target", "change"}
MASK_KEYS_SUBSTR = {"groundtruth", "ground_truth", "label1", "change_mask"}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Manual overrides -- fill these in ONLY if automatic detection fails.
# Example: {"A": "LEVIR_CD/train/A", "B": "LEVIR_CD/train/B", "mask": "LEVIR_CD/train/label"}
OVERRIDES: dict[str, str] = {}

SPLIT_NAME_MAP = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
    "eval": "test",
}


@dataclass
class Sample:
    """One matched triplet: image at t1, image at t2, change mask."""
    stem: str
    a_path: Path
    b_path: Path
    mask_path: Path
    split: str  # "train" | "val" | "test"


# ---------------------------------------------------------------------------
# Download / tree printing
# ---------------------------------------------------------------------------

def download_dataset() -> Path:
    """Download via kagglehub (no-op if already cached) and return the path."""
    # cache location decided BEFORE kagglehub is imported (it reads the env var)
    import os
    os.environ.setdefault("KAGGLEHUB_CACHE", str(config.KAGGLEHUB_CACHE))
    try:
        import kagglehub
    except ImportError:
        sys.exit(
            "ERROR: kagglehub is not installed. Run:  pip install kagglehub"
        )
    print(f"[data] Downloading / locating dataset '{config.KAGGLE_DATASET}' ...")
    print(f"[data] kagglehub cache: {os.environ['KAGGLEHUB_CACHE']}")
    path = Path(kagglehub.dataset_download(config.KAGGLE_DATASET))
    print(f"[data] Dataset root: {path}")
    if not path.exists():
        sys.exit("ERROR: kagglehub returned a path that does not exist.")
    return path


def print_tree(root: Path, max_depth: int = 4, max_names: int = 3) -> None:
    """Print the folder tree with file counts and a few sample filenames."""
    print(f"\n[data] Directory tree of: {root}")
    dirs = [root]
    depth = {root: 0}
    while dirs:
        d = dirs.pop(0)
        if depth[d] > max_depth:
            continue
        try:
            entries = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            continue
        subdirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]
        indent = "    " * depth[d]
        ext_counts = Counter(e.suffix.lower() for e in files)
        ext_str = ", ".join(f"{k or '(no ext)'}: {v}" for k, v in ext_counts.most_common(5))
        print(f"{indent}{d.name}/  [{len(files)} files | {len(subdirs)} subdirs"
              + (f" | {ext_str}" if ext_str else "") + "]")
        for f in files[:max_names]:
            print(f"{indent}    {f.name}")
        if len(files) > max_names:
            print(f"{indent}    ... and {len(files) - max_names} more")
        dirs.extend(subdirs)
        for s in subdirs:
            depth[s] = depth[d] + 1


# ---------------------------------------------------------------------------
# Classification of folders
# ---------------------------------------------------------------------------

def _classify_name(name: str) -> str | None:
    n = name.strip().lower()
    if n in A_KEYS_EXACT or any(k in n for k in A_KEYS_SUBSTR):
        return "A"
    if n in B_KEYS_EXACT or any(k in n for k in B_KEYS_SUBSTR):
        return "B"
    if n in MASK_KEYS_EXACT or any(k in n for k in MASK_KEYS_SUBSTR):
        return "mask"
    return None


def classify_dirs(root: Path) -> dict[Path, str]:
    """Return {dir_path: role} for every folder that contains images and
    whose name (or path) suggests one of the three roles."""
    roles: dict[Path, str] = {}
    for d in sorted(root.rglob("*")):
        if not d.is_dir():
            continue
        has_img = any(p.suffix.lower() in IMG_EXTS for p in d.iterdir() if p.is_file())
        if not has_img:
            continue
        # 1) own folder name
        role = _classify_name(d.name)
        # 2) fall back to the full relative path (helps with e.g. "images2")
        if role is None:
            rel = "/".join(p.lower() for p in d.relative_to(root).parts)
            role = _classify_name(rel)
        if role:
            roles[d] = role
    return roles


# ---------------------------------------------------------------------------
# Matching A / B / mask by common stem
# ---------------------------------------------------------------------------

def _stems_of(d: Path) -> dict[str, Path]:
    out = {}
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out[p.stem] = p
    return out


def _split_name_for(parent: Path) -> str | None:
    return SPLIT_NAME_MAP.get(parent.name.strip().lower())


def build_samples(root: Path) -> tuple[list[Sample], list[str]]:
    """Match A/B/mask folders and build the sample list.

    Returns (samples, notes) where `notes` documents every assumption made,
    so the user can audit the automatic detection.
    """
    notes: list[str] = []

    if OVERRIDES:
        notes.append(f"Manual overrides in use: {OVERRIDES}")
        roles = {Path(v).resolve(): k for k, v in OVERRIDES.items()}
        for p, r in roles.items():
            if not p.exists():
                sys.exit(f"ERROR: overridden {r} folder does not exist: {p}")
    else:
        roles = classify_dirs(root)

    if not roles:
        print("\n[data] Could not classify any folder automatically.")
        print_tree(root, max_depth=4)
        sys.exit(
            "ERROR: automatic folder detection failed. Inspect the tree above "
            "and fill `OVERRIDES` in src/data_discovery.py."
        )

    print("\n[data] Detected role folders:")
    for d, r in sorted(roles.items()):
        print(f"    [{r:>4}] {d}")

    # group by the A-folder's parent so that e.g.  train/A, train/B, train/label
    # are combined.  Some layouts nest mask under a sibling folder, so we also
    # try the grandparent when a parent group is incomplete.
    by_parent: dict[Path, dict[str, Path]] = {}
    for d, r in roles.items():
        by_parent.setdefault(d.parent, {}).setdefault(r, d)

    groups: list[tuple[Path, dict[str, Path]]] = []
    used: set[Path] = set()
    for parent, rd in sorted(by_parent.items()):
        if "A" in rd and "B" in rd:
            groups.append((parent, rd))
            used.add(parent)

    if not groups:
        print_tree(root, max_depth=4)
        sys.exit(
            "ERROR: no folder group contains both an A (time-1) and a B (time-2) "
            "image folder. Inspect the tree above and fill `OVERRIDES`."
        )

    # Warn about folders that could not be attached to any group
    for parent, rd in by_parent.items():
        if parent not in used:
            notes.append(f"Unmatched folder group at {parent}: roles={sorted(rd)}")

    # official splits?
    official: dict[str, list[tuple[Path, dict[str, Path]]]] = {"train": [], "val": [], "test": []}
    unnamed: list[tuple[Path, dict[str, Path]]] = []
    for parent, rd in groups:
        s = _split_name_for(parent)
        if s:
            official[s].append((parent, rd))
        else:
            unnamed.append((parent, rd))

    complete_official = all(official[s] for s in ("train", "val", "test")) and not unnamed
    if complete_official:
        notes.append(
            "Dataset ships complete official train/val/test splits -- these are "
            "used as-is (no re-splitting, avoids any leakage)."
        )
        group_splits: list[tuple[Path, dict[str, Path], str]] = [
            (p, rd, s) for s in ("train", "val", "test") for p, rd in official[s]
        ]
    else:
        # keep every official group under its own name; unnamed groups join the
        # train pool. If any split ends up empty, everything is pooled and
        # re-split below (seeded), so no samples are ever silently dropped.
        notes.append(
            "Official splits absent or incomplete -- all samples will be pooled "
            "and (if needed) re-split 70/15/15 with a fixed seed."
        )
        group_splits = [(p, rd, s) for s in ("train", "val", "test")
                        for p, rd in official[s]]
        group_splits += [(p, rd, "train") for p, rd in unnamed]

    # match stems inside each group
    raw: dict[str, list[tuple[str, Path, Path, Path]]] = {"train": [], "val": [], "test": []}
    for parent, rd, split in group_splits:
        stems_a = _stems_of(rd["A"])
        stems_b = _stems_of(rd["B"])
        stems_m = _stems_of(rd["mask"]) if "mask" in rd else None
        common = sorted(set(stems_a) & set(stems_b))
        if stems_m is not None:
            common = [s for s in common if s in stems_m]
        n_only_a = len(set(stems_a) - set(stems_b))
        n_only_b = len(set(stems_b) - set(stems_a))
        if n_only_a or n_only_b:
            notes.append(
                f"'{parent.name}': {n_only_a} A-file(s) without B counterpart, "
                f"{n_only_b} B-file(s) without A counterpart (skipped)."
            )
        if stems_m is None:
            sys.exit(f"ERROR: no change-mask folder found next to '{parent}'. "
                     "Fill `OVERRIDES` in src/data_discovery.py.")
        for s in common:
            raw[split].append((s, stems_a[s], stems_b[s], stems_m[s]))

    all_pairs = [x for v in raw.values() for x in v]
    if not all_pairs:
        sys.exit("ERROR: stem matching produced 0 samples. Check the printed folder "
                 "roles and filenames, then adjust `OVERRIDES` / matching rules.")

    # duplicates guard (same stem used twice within a split)
    for split, items in raw.items():
        c = Counter(s for s, *_ in items)
        dups = [k for k, v in c.items() if v > 1]
        if dups:
            sys.exit(f"ERROR: duplicate stems in '{split}': {dups[:5]} ...")

    # re-split when the official splits are absent or incomplete (e.g. this
    # dataset ships only train/ and test/ with no val/): pool every matched
    # sample and do a seeded 70/15/15 split so all three splits exist.
    if not complete_official or not all(raw[s] for s in ("train", "val", "test")):
        counts = {s: len(raw[s]) for s in ("train", "val", "test")}
        notes.append(
            f"Official splits incomplete (found {counts}) -- pooling ALL samples "
            f"and re-splitting {config.TRAIN_RATIO:.0%}/{config.VAL_RATIO:.0%}/"
            f"{config.TEST_RATIO:.0%} with seed={config.SEED}."
        )
        stems = sorted(set(s for s, *_ in all_pairs))
        rng = random.Random(config.SEED)
        rng.shuffle(stems)
        n = len(stems)
        n_tr = int(n * config.TRAIN_RATIO)
        n_va = int(n * config.VAL_RATIO)
        cut = {"train": set(stems[:n_tr]),
               "val": set(stems[n_tr:n_tr + n_va]),
               "test": set(stems[n_tr + n_va:])}
        raw = {"train": [], "val": [], "test": []}
        for s, a, b, m in all_pairs:
            for sp, keep in cut.items():
                if s in keep:
                    raw[sp].append((s, a, b, m))
        notes.append(f"Re-split sizes: train={len(cut['train'])}, "
                     f"val={len(cut['val'])}, test={len(cut['test'])} "
                     f"(seed={config.SEED}).")

    samples: list[Sample] = []
    for split in ("train", "val", "test"):
        for s, a, b, m in sorted(raw[split], key=lambda t: t[0]):
            samples.append(Sample(stem=s, a_path=a, b_path=b, mask_path=m, split=split))

    # leakage check between splits
    sets = {sp: {x.stem for x in samples if x.split == sp} for sp in ("train", "val", "test")}
    for s1 in ("train", "val", "test"):
        for s2 in ("train", "val", "test"):
            if s1 < s2:
                inter = sets[s1] & sets[s2]
                if inter:
                    sys.exit(f"ERROR: leakage between '{s1}' and '{s2}': {sorted(inter)[:5]}")
    notes.append("Leakage check passed: no sample id appears in two splits.")

    return samples, notes


# ---------------------------------------------------------------------------
# Verification & statistics
# ---------------------------------------------------------------------------

def load_mask(path: Path) -> np.ndarray:
    """Load a change mask as float32 in {0,1}, robust to {0,1}, {0,255}, {0,128}."""
    arr = np.array(Image.open(path).convert("L"))
    if arr.max() == 0:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr > arr.max() / 2.0).astype(np.float32)


def verify_samples(samples: list[Sample], check_n: int = 30) -> dict:
    """Check the first `check_n` samples (plus the last one) for consistency."""
    idxs = list(range(min(check_n, len(samples)))) + [len(samples) - 1]
    sizes: Counter = Counter()
    mask_values: set[int] = set()
    chan: Counter = Counter()
    problems: list[str] = []
    for i in idxs:
        s = samples[i]
        if not (s.a_path.exists() and s.b_path.exists() and s.mask_path.exists()):
            problems.append(f"missing file(s) in sample '{s.stem}'")
            continue
        ia, ib = Image.open(s.a_path), Image.open(s.b_path)
        m = load_mask(s.mask_path)
        if ia.size != ib.size:
            problems.append(f"'{s.stem}': A size {ia.size} != B size {ib.size}")
        if ia.size != (m.shape[1], m.shape[0]):
            problems.append(f"'{s.stem}': mask {m.shape[::-1]} != image {ia.size}")
        sizes[ia.size] += 1
        chan[ia.mode] += 1
        mask_values.update(np.unique(m).astype(int).tolist())
    return {"sizes": dict(sizes), "image_modes": dict(chan),
            "mask_unique_values": sorted(mask_values), "problems": problems}


def change_ratio_stats(samples: list[Sample], max_n: int | None = None) -> dict:
    """Fraction of changed pixels (mean over sampled masks)."""
    rng = random.Random(config.SEED)
    pool = samples[:]
    rng.shuffle(pool)
    if max_n:
        pool = pool[:max_n]
    ratios = []
    for s in pool:
        m = load_mask(s.mask_path)
        ratios.append(float(m.mean()))
    return {
        "n_masks": len(ratios),
        "changed_frac_mean": float(np.mean(ratios)) if ratios else 0.0,
        "unchanged_frac_mean": float(1.0 - np.mean(ratios)) if ratios else 1.0,
        "changed_frac_min": float(np.min(ratios)) if ratios else 0.0,
        "changed_frac_max": float(np.max(ratios)) if ratios else 0.0,
        "empty_masks": int(sum(1 for r in ratios if r == 0.0)),
    }


def split_report(samples: list[Sample]) -> dict:
    """Counts + change-pixel statistics per split (also used by train.py)."""
    report: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        sub = [s for s in samples if s.split == split]
        stats = change_ratio_stats(sub, max_n=config.MAX_MASKS_FOR_STATS)
        report[split] = {"n_samples": len(sub), **stats}
    return report
