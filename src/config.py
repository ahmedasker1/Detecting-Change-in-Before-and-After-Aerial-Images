# -*- coding: utf-8 -*-
"""Central configuration for the LEVIR-CD change-detection project.

Every tunable knob lives here (or is overridable from the command line in
train.py / evaluate.py). Paths are relative to the project root
(the directory that contains the `src/` folder).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
LOGS_DIR = OUTPUT_DIR / "logs"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
KAGGLE_DATASET = "mdrifaturrahman33/levir-cd-change-detection"

# Where kagglehub stores the downloaded dataset. Defaults to a cache folder
# next to the project (so it lands on the project drive) -- can be overridden
# with the KAGGLEHUB_CACHE environment variable.
KAGGLEHUB_CACHE = Path(
    os.environ.get("KAGGLEHUB_CACHE", str(PROJECT_ROOT / ".kagglehub_cache"))
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Splits (used ONLY when the dataset does not ship official splits)
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------------------------------------------------------------------
# Images / patches
# ---------------------------------------------------------------------------
# Training crops (and eval tiles) are IMAGE_SIZE x IMAGE_SIZE. Images bigger
# than this are split into a grid of tiles; smaller ones are reflect-padded.
IMAGE_SIZE = 256          # must be a multiple of 16 (4 downsampling stages)
MAX_MASKS_FOR_STATS = 300  # how many masks to sample for change-ratio stats

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
IN_CHANNELS = 3
BASE_CHANNELS = 32        # U-Net width; 32 fits a 4 GB GPU comfortably

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
NUM_WORKERS = 2
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BCE_WEIGHT = 0.5          # weight of BCEWithLogitsLoss in the combined loss
DICE_WEIGHT = 0.5         # weight of soft-Dice loss in the combined loss
# Class imbalance option: BCE pos_weight = neg/pos ratio. NOTE: default OFF --
# with correctly-measured validation (see the double-sigmoid bugfix note in
# train.py) plain BCE+Dice trains fine on this dataset; enable for experiments.
USE_AUTO_POS_WEIGHT = False
MAX_POS_WEIGHT = 20.0
EARLY_STOP_PATIENCE = 15  # epochs without val-IoU improvement before stopping
LR_SCHEDULER_PATIENCE = 5 # ReduceLROnPlateau patience (on val IoU)
LR_SCHEDULER_FACTOR = 0.5
USE_AMP = True            # mixed precision on GPU
GRAD_CLIP = 1.0

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
THRESHOLD = 0.5           # sigmoid(logits) > THRESHOLD  ->  change pixel
NUM_VIS_SAMPLES = 6       # qualitative figures saved by evaluate.py

# ---------------------------------------------------------------------------
# Augmentation (synchronised across A / B / mask, so alignment is preserved)
# ---------------------------------------------------------------------------
AUG_HFLIP = True
AUG_VFLIP = True
AUG_ROT90 = True          # random k*90 degrees (exact, keeps masks binary)


def ensure_dirs() -> None:
    """Create the standard output folders if missing."""
    for d in (OUTPUT_DIR, FIGURES_DIR, CHECKPOINTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def as_dict() -> dict:
    """All config values as a plain dict (for saving with a run)."""
    return {
        "kaggle_dataset": KAGGLE_DATASET,
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "image_size": IMAGE_SIZE,
        "in_channels": IN_CHANNELS,
        "base_channels": BASE_CHANNELS,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "bce_weight": BCE_WEIGHT,
        "dice_weight": DICE_WEIGHT,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "lr_scheduler_patience": LR_SCHEDULER_PATIENCE,
        "lr_scheduler_factor": LR_SCHEDULER_FACTOR,
        "use_amp": USE_AMP,
        "grad_clip": GRAD_CLIP,
        "threshold": THRESHOLD,
        "aug": {
            "hflip": AUG_HFLIP,
            "vflip": AUG_VFLIP,
            "rot90": AUG_ROT90,
        },
    }
