"""Utilities for binary dataset creation, balancing, and patient-level splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src import config


REQUIRED_METADATA_COLUMNS = {
    "Image Index",
    "Finding Labels",
    "Patient ID",
    "Patient Age",
    "Patient Gender",
    "View Position",
}


def load_metadata(csv_path: Path | None = None) -> pd.DataFrame:
    """Load NIH metadata and validate required columns."""
    csv_path = csv_path or (config.RAW_DATA_DIR / config.DATA_ENTRY_FILENAME)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {csv_path}. "
            "Run notebooks/00_download_dataset_kaggle.ipynb first."
        )

    df = pd.read_csv(csv_path)
    missing_columns = REQUIRED_METADATA_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required metadata columns: {sorted(missing_columns)}")

    return df


def add_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add reusable label flags for the binary CardioIA task."""
    result = df.copy()
    result["labels_list"] = result["Finding Labels"].astype(str).str.split("|")
    result["num_labels"] = result["labels_list"].apply(len)
    result["is_no_finding"] = result["Finding Labels"].eq(config.NEGATIVE_LABEL)
    result["has_cardiomegaly"] = result["labels_list"].apply(
        lambda labels: config.TARGET_LABEL in labels
    )
    result["is_cardiomegaly_only"] = result["Finding Labels"].eq(config.TARGET_LABEL)
    result["is_multilabel"] = result["num_labels"] > 1
    return result


def attach_image_paths(
    df: pd.DataFrame, image_index_path: Path | None = None
) -> pd.DataFrame:
    """Attach local image paths when data/raw/image_paths.csv exists."""
    image_index_path = image_index_path or (
        config.RAW_DATA_DIR / config.IMAGE_PATHS_FILENAME
    )
    if not image_index_path.exists():
        result = df.copy()
        result["relative_path"] = pd.NA
        result["absolute_path"] = pd.NA
        result["image_file_exists"] = False
        return result

    image_index = pd.read_csv(image_index_path)
    expected_columns = {"image_name", "relative_path", "absolute_path", "exists"}
    missing_columns = expected_columns - set(image_index.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns in {image_index_path}: {sorted(missing_columns)}"
        )

    image_index = image_index.rename(
        columns={
            "image_name": "Image Index",
            "exists": "image_file_exists",
        }
    )
    return df.merge(
        image_index[
            ["Image Index", "relative_path", "absolute_path", "image_file_exists"]
        ],
        on="Image Index",
        how="left",
    )


def _finalize_binary_dataset(df: pd.DataFrame, dataset_version: str) -> pd.DataFrame:
    result = df.copy()
    result["binary_label"] = np.where(result["has_cardiomegaly"], 1, 0).astype(int)
    result["binary_label_name"] = result["binary_label"].map(config.INDEX_TO_CLASS)
    result["dataset_version"] = dataset_version

    preferred_columns = [
        "Image Index",
        "Finding Labels",
        "labels_list",
        "num_labels",
        "is_multilabel",
        "Patient ID",
        "Patient Age",
        "Patient Gender",
        "View Position",
        "OriginalImageWidth",
        "OriginalImageHeight",
        "binary_label",
        "binary_label_name",
        "dataset_version",
        "relative_path",
        "absolute_path",
        "image_file_exists",
    ]
    existing_columns = [column for column in preferred_columns if column in result.columns]
    remaining_columns = [column for column in result.columns if column not in existing_columns]
    return result[existing_columns + remaining_columns].reset_index(drop=True)


def create_binary_datasets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create clean and realistic No Finding vs Cardiomegaly datasets."""
    with_labels = add_label_columns(df)

    clean = with_labels[with_labels["is_no_finding"] | with_labels["is_cardiomegaly_only"]]
    realistic = with_labels[with_labels["is_no_finding"] | with_labels["has_cardiomegaly"]]

    clean = _finalize_binary_dataset(clean, "clean")
    realistic = _finalize_binary_dataset(realistic, "realistic")
    return clean, realistic


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return class counts and percentages for a binary dataset."""
    distribution = (
        df.groupby(["binary_label", "binary_label_name"], dropna=False)
        .size()
        .reset_index(name="image_count")
        .sort_values("binary_label")
    )
    distribution["percentage"] = distribution["image_count"] / len(df)
    return distribution


def undersample_balance(
    df: pd.DataFrame, seed: int = config.SEED
) -> pd.DataFrame:
    """Balance classes by undersampling every class to the minority count."""
    class_counts = df["binary_label"].value_counts()
    if class_counts.empty:
        raise ValueError("Cannot balance an empty dataset.")
    target_count = int(class_counts.min())

    balanced_parts = []
    for label, group in df.groupby("binary_label", group_keys=False):
        balanced_parts.append(group.sample(n=target_count, random_state=seed))

    balanced = (
        pd.concat(balanced_parts, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    balanced["balance_strategy"] = "undersampling"
    return balanced


def compute_class_weights(df: pd.DataFrame) -> dict[str, float]:
    """Compute inverse-frequency class weights normalized around 1.0."""
    counts = df["binary_label"].value_counts().sort_index()
    if counts.empty:
        raise ValueError("Cannot compute class weights for an empty dataset.")

    total = counts.sum()
    num_classes = len(counts)
    weights = total / (num_classes * counts)
    return {str(int(label)): float(weight) for label, weight in weights.items()}


def save_class_weights(weights: dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(weights, indent=2, sort_keys=True) + "\n")


def split_patient_ids(
    df: pd.DataFrame,
    seed: int = config.SEED,
    validation_size: float = config.VALIDATION_SIZE,
    test_size: float = config.TEST_SIZE,
) -> dict[str, set[Any]]:
    """Split patient IDs with approximate stratification by patient-level positivity."""
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must be > 0 and sum to < 1.")

    patient_summary = (
        df.groupby("Patient ID")
        .agg(
            patient_positive=("binary_label", "max"),
            image_count=("Image Index", "count"),
        )
        .reset_index()
    )

    rng = np.random.default_rng(seed)
    splits = {"train": set(), "val": set(), "test": set()}

    for _, group in patient_summary.groupby("patient_positive"):
        patient_ids = group["Patient ID"].to_numpy()
        rng.shuffle(patient_ids)

        n_total = len(patient_ids)
        n_test = int(round(n_total * test_size))
        n_val = int(round(n_total * validation_size))

        test_ids = patient_ids[:n_test]
        val_ids = patient_ids[n_test : n_test + n_val]
        train_ids = patient_ids[n_test + n_val :]

        splits["test"].update(test_ids.tolist())
        splits["val"].update(val_ids.tolist())
        splits["train"].update(train_ids.tolist())

    return splits


def apply_patient_split(
    df: pd.DataFrame,
    patient_splits: dict[str, set[Any]],
) -> dict[str, pd.DataFrame]:
    """Return train, validation, and test DataFrames from patient ID splits."""
    output = {}
    for split_name, patient_ids in patient_splits.items():
        split_df = df[df["Patient ID"].isin(patient_ids)].copy().reset_index(drop=True)
        split_df["split"] = split_name
        output[split_name] = split_df
    return output


def validate_patient_split(split_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Validate patient leakage and summarize split composition."""
    split_names = list(split_frames)
    patient_sets = {
        name: set(frame["Patient ID"].unique())
        for name, frame in split_frames.items()
    }

    leakage_pairs = []
    for i, left_name in enumerate(split_names):
        for right_name in split_names[i + 1 :]:
            overlap = patient_sets[left_name] & patient_sets[right_name]
            leakage_pairs.append(
                {
                    "left_split": left_name,
                    "right_split": right_name,
                    "overlapping_patients": len(overlap),
                }
            )

    rows = []
    for split_name, frame in split_frames.items():
        distribution = class_distribution(frame)
        counts = {
            int(row["binary_label"]): int(row["image_count"])
            for _, row in distribution.iterrows()
        }
        rows.append(
            {
                "split": split_name,
                "image_count": len(frame),
                "patient_count": frame["Patient ID"].nunique(),
                "negative_count": counts.get(0, 0),
                "positive_count": counts.get(1, 0),
                "positive_rate": counts.get(1, 0) / len(frame) if len(frame) else 0,
            }
        )

    summary = pd.DataFrame(rows)
    leakage = pd.DataFrame(leakage_pairs)
    summary["has_patient_leakage"] = bool(
        not leakage.empty and leakage["overlapping_patients"].sum() > 0
    )
    return summary


def save_split_frames(
    split_frames: dict[str, pd.DataFrame],
    output_dir: Path,
    prefix: str | None = None,
) -> dict[str, Path]:
    """Save split DataFrames and return output paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split_name, frame in split_frames.items():
        filename = f"{split_name}.csv" if prefix is None else f"{split_name}_{prefix}.csv"
        path = output_dir / filename
        frame.to_csv(path, index=False)
        paths[split_name] = path
    return paths


def write_summary_table(
    tables: Iterable[tuple[str, pd.DataFrame]], output_path: Path
) -> pd.DataFrame:
    """Combine named tables with a source column and save them."""
    combined = []
    for name, table in tables:
        table_copy = table.copy()
        table_copy.insert(0, "source", name)
        combined.append(table_copy)

    result = pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result
