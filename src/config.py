"""Global configuration for the CardioIA Vision project.

This module keeps paths and experiment defaults in one place so notebooks and
scripts use the same project layout.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCS_DIR = PROJECT_ROOT / "Docs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_IMAGES_DIR = RAW_DATA_DIR / "images"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figuras"
METRICS_DIR = REPORTS_DIR / "metricas"
TABLES_DIR = REPORTS_DIR / "tabelas"

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
EXPORTED_MODELS_DIR = MODELS_DIR / "exported"

APP_DIR = PROJECT_ROOT / "src" / "app"
UPLOADS_DIR = APP_DIR / "static" / "uploads"

DATASET_SLUG = "nih-chest-xrays/data"
DATASET_URL = "https://www.kaggle.com/datasets/nih-chest-xrays/data"
DATA_ENTRY_FILENAME = "Data_entry_2017.csv"
IMAGE_PATHS_FILENAME = "image_paths.csv"

TARGET_LABEL = "Cardiomegaly"
NEGATIVE_LABEL = "No Finding"
CLASS_NAMES = [NEGATIVE_LABEL, TARGET_LABEL]
CLASS_TO_INDEX = {NEGATIVE_LABEL: 0, TARGET_LABEL: 1}
INDEX_TO_CLASS = {index: label for label, index in CLASS_TO_INDEX.items()}

IMAGE_SIZE = 224
IMAGE_CHANNELS = 3
BATCH_SIZE = 32
NUM_WORKERS = 4
SEED = 42
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

QUICK_TEST_EPOCHS = 1
DEFAULT_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

MODEL_METRIC_TO_MAXIMIZE = "f1"
DEFAULT_THRESHOLD = 0.5


def get_device() -> str:
    """Return the preferred training device without requiring torch at import time."""
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


DEVICE = get_device()


def ensure_project_directories() -> None:
    """Create all expected project directories if they do not exist."""
    directories = [
        RAW_DATA_DIR,
        RAW_IMAGES_DIR,
        PROCESSED_DATA_DIR,
        SPLITS_DIR,
        NOTEBOOKS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        TABLES_DIR,
        CHECKPOINTS_DIR,
        EXPORTED_MODELS_DIR,
        UPLOADS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = SEED) -> None:
    """Set common random seeds for reproducible experiments."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def as_dict() -> dict[str, Any]:
    """Return key configuration values for notebook display and logging."""
    return {
        "project_root": str(PROJECT_ROOT),
        "dataset_slug": DATASET_SLUG,
        "dataset_url": DATASET_URL,
        "raw_data_dir": str(RAW_DATA_DIR),
        "raw_images_dir": str(RAW_IMAGES_DIR),
        "splits_dir": str(SPLITS_DIR),
        "reports_dir": str(REPORTS_DIR),
        "checkpoints_dir": str(CHECKPOINTS_DIR),
        "exported_models_dir": str(EXPORTED_MODELS_DIR),
        "target_label": TARGET_LABEL,
        "negative_label": NEGATIVE_LABEL,
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "seed": SEED,
        "device": DEVICE,
        "default_epochs": DEFAULT_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "default_threshold": DEFAULT_THRESHOLD,
    }

