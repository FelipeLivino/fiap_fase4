"""Utilities for consolidating model metrics and selecting the final model."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src import config


METRIC_COLUMNS = [
    "model_name",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc_roc",
    "seconds_per_image",
    "total_parameters",
    "trainable_parameters",
    "checkpoint_size_bytes",
    "checkpoint_path",
]


def load_metric_files(metrics_dir: Path | None = None) -> pd.DataFrame:
    """Load all per-model `*_metrics.csv` files into one DataFrame."""
    metrics_dir = metrics_dir or config.METRICS_DIR
    metric_files = sorted(metrics_dir.glob("*_metrics.csv"))
    frames = []

    for metric_file in metric_files:
        # Skip consolidated outputs to avoid duplicating rows on reruns.
        if metric_file.name in {
            "model_comparison_all_metrics.csv",
            "model_comparison_ranked.csv",
            "final_model_selection.csv",
        }:
            continue
        frame = pd.read_csv(metric_file)
        frame["source_file"] = metric_file.name
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=METRIC_COLUMNS + ["source_file"])

    combined = pd.concat(frames, ignore_index=True)
    for column in METRIC_COLUMNS:
        if column not in combined.columns:
            combined[column] = pd.NA
    return combined


def add_selection_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add ranking fields for final model selection."""
    if metrics.empty:
        return metrics.copy()

    result = metrics.copy()
    numeric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc_roc",
        "seconds_per_image",
        "total_parameters",
        "trainable_parameters",
        "checkpoint_size_bytes",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result["rank_f1"] = result["f1"].rank(ascending=False, method="min")
    result["rank_recall"] = result["recall"].rank(ascending=False, method="min")
    result["rank_auc"] = result["auc_roc"].rank(ascending=False, method="min")
    result["rank_inference"] = result["seconds_per_image"].rank(ascending=True, method="min")

    # Health-oriented choice: prioritize F1 and recall, then AUC, then inference cost.
    result["selection_score"] = (
        result["rank_f1"] * 0.40
        + result["rank_recall"] * 0.30
        + result["rank_auc"] * 0.20
        + result["rank_inference"] * 0.10
    )
    return result.sort_values(
        ["selection_score", "rank_f1", "rank_recall", "rank_auc", "rank_inference"],
        ascending=True,
    ).reset_index(drop=True)


def choose_final_model(metrics: pd.DataFrame) -> dict[str, Any] | None:
    """Choose final model from ranked metrics."""
    ranked = add_selection_scores(metrics)
    if ranked.empty:
        return None
    return ranked.iloc[0].to_dict()


def export_final_checkpoint(
    selected_model: dict[str, Any],
    output_dir: Path | None = None,
) -> Path | None:
    """Copy the selected checkpoint to models/exported when available."""
    output_dir = output_dir or config.EXPORTED_MODELS_DIR
    checkpoint_path = selected_model.get("checkpoint_path")
    if checkpoint_path is None or pd.isna(checkpoint_path):
        return None

    source = Path(str(checkpoint_path))
    if not source.exists():
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = str(selected_model.get("model_name", "final_model"))
    destination = output_dir / f"{model_name}_final.pt"
    shutil.copy2(source, destination)
    return destination
