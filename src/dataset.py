# -*- coding: utf-8 -*-
"""Torch datasets for bi-temporal change detection.

- `TrainChangeDataset`: one random IMAGE_SIZE crop per sample per epoch,
  with synchronised flips / k*90 rotations applied identically to A, B and
  mask (so spatial alignment is never broken and masks stay binary).
- `EvalTileDataset`: deterministic non-overlapping IMAGE_SIZE tiles covering
  every image (images are reflect-padded when their size is not a multiple
  of IMAGE_SIZE). Each item carries its origin so predictions can be
  re-assembled into full images and padded regions can be excluded from
  metric accumulation.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data_discovery import Sample, load_mask


# ---------------------------------------------------------------------------
# Loading / preprocessing
# ---------------------------------------------------------------------------

def load_image_rgb(path, mean=0.5, std=0.5) -> np.ndarray:
    """RGB image as float32 (C, H, W), normalised with mean/std."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)  # HWC -> CHW


# ---------------------------------------------------------------------------
# Synchronised geometric augmentation (train only)
# ---------------------------------------------------------------------------

def _pad_to_at_least(arr: np.ndarray, th: int, tw: int) -> np.ndarray:
    """Reflect-pad (C,H,W) or (H,W) so H>=th and W>=tw."""
    h, w = arr.shape[-2], arr.shape[-1]
    ph, pw = max(0, th - h), max(0, tw - w)
    if ph == 0 and pw == 0:
        return arr
    if arr.ndim == 3:
        return np.pad(arr, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    return np.pad(arr, ((0, ph), (0, pw)), mode="reflect")


def train_transform(a: np.ndarray, b: np.ndarray, m: np.ndarray,
                    size: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random crop + synchronised flips / rot90 applied to (a, b, m)."""
    # random crop, same window for all three arrays
    a = _pad_to_at_least(a, size, size)
    b = _pad_to_at_least(b, size, size)
    m = _pad_to_at_least(m, size, size)
    h, w = m.shape
    top = rng.randint(0, h - size)
    left = rng.randint(0, w - size)
    sl = (slice(top, top + size), slice(left, left + size))
    a, b, m = a[..., sl[0], sl[1]], b[..., sl[0], sl[1]], m[sl[0], sl[1]]
    # synchronised geometric flips (exact -> masks stay binary)
    if rng.random() < 0.5:  # horizontal flip (last axis = width)
        a, b, m = a[..., ::-1], b[..., ::-1], m[:, ::-1]
    if rng.random() < 0.5:  # vertical flip (second-to-last axis = height)
        a, b, m = a[..., ::-1, :], b[..., ::-1, :], m[::-1, :]
    k = rng.randint(0, 3)   # k * 90 degrees
    if k:
        a, b, m = np.rot90(a, k, axes=(-2, -1)), np.rot90(b, k, axes=(-2, -1)), np.rot90(m, k)
    return np.ascontiguousarray(a), np.ascontiguousarray(b), np.ascontiguousarray(m)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TrainChangeDataset(Dataset):
    def __init__(self, samples: list[Sample], image_size: int = 256,
                 augment: bool = True):
        self.samples = samples
        self.size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        a = load_image_rgb(s.a_path)
        b = load_image_rgb(s.b_path)
        m = load_mask(s.mask_path)
        if self.augment:
            # `random` is seeded per DataLoader worker by torch, so each worker
            # produces a different (but reproducible) augmentation stream.
            rng = random.Random(random.getrandbits(63))
            a, b, m = train_transform(a, b, m, self.size, rng)
        return {
            "a": torch.from_numpy(a.copy()),
            "b": torch.from_numpy(b.copy()),
            "mask": torch.from_numpy(m.copy()).unsqueeze(0),
        }


def tile_meta(sample_idx: int, stem: str, top: int, left: int,
              valid_h: int, valid_w: int) -> dict:
    """Per-tile metadata as a plain dict (collates to tensors in DataLoaders)."""
    return {"sample_idx": sample_idx, "stem": stem, "top": top, "left": left,
            "valid_h": valid_h,   # tile rows that belong to the real image
            "valid_w": valid_w}   # tile cols that belong to the real image


class EvalTileDataset(Dataset):
    """All IMAGE_SIZE tiles of every sample, in a deterministic order."""

    def __init__(self, samples: list[Sample], image_size: int = 256,
                 cache_images: int = 8):
        self.samples = samples
        self.size = image_size
        self.tiles: list[tuple[int, int, int]] = []  # (sample_idx, top, left)
        for i, s in enumerate(samples):
            img = Image.open(s.a_path)
            h, w = img.size[1], img.size[0]
            for top in range(0, h, image_size):
                for left in range(0, w, image_size):
                    self.tiles.append((i, top, left))
        self._cache: dict[int, tuple] = {}
        self._cache_order: list[int] = []
        self._cache_max = max(1, cache_images)

    def __len__(self):
        return len(self.tiles)

    def _load_sample(self, i: int):
        if i not in self._cache:
            s = self.samples[i]
            a = load_image_rgb(s.a_path)
            b = load_image_rgb(s.b_path)
            m = load_mask(s.mask_path)
            self._cache[i] = (a, b, m)
            self._cache_order.append(i)
            # bounded FIFO cache: keeps per-worker RAM usage predictable
            while len(self._cache_order) > self._cache_max:
                self._cache.pop(self._cache_order.pop(0), None)
        return self._cache[i]

    def __getitem__(self, t):
        i, top, left = self.tiles[t]
        a, b, m = self._load_sample(i)
        s = self.samples[i]
        size = self.size
        th, tw = a.shape[-2], a.shape[-1]
        vh, vw = min(size, th - top), min(size, tw - left)
        ph, pw = size - vh, size - vw

        def tile_pad(x):
            x = x[..., top:top + vh, left:left + vw]
            if ph or pw:  # reflect-pad bottom/right partial tiles
                x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect")
            return x

        ta, tb = tile_pad(a), tile_pad(b)
        tm = m[top:top + vh, left:left + vw]
        if ph or pw:
            tm = np.pad(tm, ((0, ph), (0, pw)), mode="constant", constant_values=0)
        meta = tile_meta(sample_idx=i, stem=s.stem, top=top, left=left,
                         valid_h=vh, valid_w=vw)
        return {
            "a": torch.from_numpy(np.ascontiguousarray(ta)),
            "b": torch.from_numpy(np.ascontiguousarray(tb)),
            "mask": torch.from_numpy(tm).unsqueeze(0),
            "meta": meta,
        }


def samples_for_split(samples: list[Sample], split: str) -> list[Sample]:
    return [s for s in samples if s.split == split]
