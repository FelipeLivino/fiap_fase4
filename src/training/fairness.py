"""Governance and subgroup analysis utilities for CardioIA Vision."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src import config
from src.training.evaluate import _logits_to_probabilities
from src.training.metrics import compute_binary_metrics


AGE_BINS = [0, 18, 40, 60, 80, 200]
AGE_LABELS = ["0-17", "18-39", "40-59", "60-79", "80+"]


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Add an age_group column using fixed clinical-style age bands."""
    result = df.copy()
    ages = pd.to_numeric(result["Patient Age"], errors="coerce")
    result["age_group"] = pd.cut(
        ages,
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False,
        include_lowest=True,
    )
    result["age_group"] = result["age_group"].astype("string").fillna("Unknown")
    return result


def representation_table(
    df: pd.DataFrame,
    group_col: str,
    label_col: str = "binary_label",
) -> pd.DataFrame:
    """Summarize representation by subgroup and class."""
    if group_col not in df.columns:
        raise ValueError(f"Column not found for representation analysis: {group_col}")

    data = df.copy()
    data[group_col] = data[group_col].astype("string").fillna("Unknown")
    grouped = (
        data.groupby([group_col, label_col], dropna=False)
        .size()
        .reset_index(name="image_count")
    )
    grouped["group_total"] = grouped.groupby(group_col)["image_count"].transform("sum")
    grouped["dataset_total"] = len(data)
    grouped["percentage_within_group"] = grouped["image_count"] / grouped["group_total"]
    grouped["percentage_of_dataset"] = grouped["image_count"] / grouped["dataset_total"]
    return grouped.sort_values([group_col, label_col]).reset_index(drop=True)


def subgroup_metric_table(
    predictions: pd.DataFrame,
    group_col: str,
    min_samples: int = 10,
    threshold: float = config.DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Compute binary metrics for each subgroup."""
    required = {"y_true", "y_prob", group_col}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing columns for subgroup metrics: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    data = predictions.copy()
    data[group_col] = data[group_col].astype("string").fillna("Unknown")

    for group_value, group_df in data.groupby(group_col, dropna=False):
        y_true = group_df["y_true"].to_numpy()
        y_prob = group_df["y_prob"].to_numpy()

        if len(group_df) < min_samples:
            metric_values = {
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "auc_roc": np.nan,
                "tn": np.nan,
                "fp": np.nan,
                "fn": np.nan,
                "tp": np.nan,
            }
            skipped = True
        else:
            metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=threshold)
            metric_values = metrics.as_dict()
            skipped = False

        rows.append(
            {
                "group_column": group_col,
                "group_value": str(group_value),
                "sample_count": len(group_df),
                "positive_count": int(group_df["y_true"].sum()),
                "positive_rate": float(group_df["y_true"].mean()) if len(group_df) else np.nan,
                "skipped_low_sample": skipped,
                **metric_values,
            }
        )

    return pd.DataFrame(rows).sort_values("sample_count", ascending=False).reset_index(drop=True)


def subgroup_gaps(metric_table: pd.DataFrame) -> pd.DataFrame:
    """Compute max-min subgroup gaps for key metrics."""
    key_metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    rows = []
    for group_col, group_df in metric_table.groupby("group_column"):
        valid = group_df[group_df["skipped_low_sample"] == False]  # noqa: E712
        for metric in key_metrics:
            values = pd.to_numeric(valid[metric], errors="coerce").dropna()
            if values.empty:
                gap = np.nan
                min_value = np.nan
                max_value = np.nan
            else:
                min_value = float(values.min())
                max_value = float(values.max())
                gap = max_value - min_value
            rows.append(
                {
                    "group_column": group_col,
                    "metric": metric,
                    "min_value": min_value,
                    "max_value": max_value,
                    "gap": gap,
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def collect_predictions_with_metadata(
    model: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device = config.DEVICE,
) -> pd.DataFrame:
    """Collect predictions and metadata from a DataLoader with return_metadata=True."""
    model.eval()
    model.to(device)
    rows: list[dict[str, Any]] = []

    for batch in dataloader:
        if not isinstance(batch, dict):
            raise TypeError(
                "Expected a dict batch. Create dataloaders with return_metadata=True."
            )

        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(images)
        probabilities = _logits_to_probabilities(logits).detach().cpu().numpy()
        y_true = labels.detach().cpu().numpy().astype(int)

        batch_size = len(y_true)
        for index in range(batch_size):
            row = {
                "image_index": batch["image_index"][index],
                "patient_id": batch["patient_id"][index],
                "finding_labels": batch["finding_labels"][index],
                "binary_label_name": batch["binary_label_name"][index],
                "image_path": batch["image_path"][index],
                "y_true": int(y_true[index]),
                "y_prob": float(probabilities[index]),
                "y_pred": int(probabilities[index] >= config.DEFAULT_THRESHOLD),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def merge_predictions_with_split_metadata(
    predictions: pd.DataFrame,
    split_csv: Path,
) -> pd.DataFrame:
    """Attach demographic and acquisition metadata from a split CSV."""
    split_df = pd.read_csv(split_csv)
    metadata_cols = [
        "Image Index",
        "Patient ID",
        "Patient Age",
        "Patient Gender",
        "View Position",
        "Finding Labels",
        "binary_label",
        "binary_label_name",
    ]
    available_cols = [column for column in metadata_cols if column in split_df.columns]
    metadata = split_df[available_cols].rename(
        columns={
            "Image Index": "image_index",
            "Patient ID": "patient_id",
            "Patient Age": "Patient Age",
            "Patient Gender": "Patient Gender",
            "View Position": "View Position",
        }
    )
    merged = predictions.merge(metadata, on=["image_index", "patient_id"], how="left")
    return add_age_group(merged)
