"""Evaluation utilities for CardioIA Vision models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src import config
from src.training.experiment_logger import save_json
from src.training.metrics import compute_binary_metrics, compute_roc_curve


def count_model_parameters(model: nn.Module) -> dict[str, int]:
    """Count total and trainable model parameters."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def _logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert model outputs to positive-class probabilities."""
    if logits.ndim == 2 and logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, 1]
    if logits.ndim == 2 and logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0])
    if logits.ndim == 1:
        return torch.sigmoid(logits)
    raise ValueError(f"Unsupported model output shape: {tuple(logits.shape)}")


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device = config.DEVICE,
) -> dict[str, np.ndarray | float]:
    """Run inference and collect labels, probabilities, and timing."""
    model.eval()
    model.to(device)

    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    total_samples = 0

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for batch in dataloader:
        if isinstance(batch, dict):
            images = batch["image"]
            labels = batch["label"]
        else:
            images, labels = batch

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probabilities = _logits_to_probabilities(logits)

        all_labels.append(labels.detach().cpu().numpy())
        all_probabilities.append(probabilities.detach().cpu().numpy())
        total_samples += int(labels.numel())

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start_time

    return {
        "y_true": np.concatenate(all_labels).astype(int),
        "y_prob": np.concatenate(all_probabilities).astype(float),
        "elapsed_seconds": float(elapsed_seconds),
        "samples": int(total_samples),
        "seconds_per_image": float(elapsed_seconds / total_samples) if total_samples else 0.0,
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    model_name: str,
    device: str | torch.device = config.DEVICE,
    threshold: float = config.DEFAULT_THRESHOLD,
    output_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a model and optionally save metrics/figures."""
    predictions = collect_predictions(model=model, dataloader=dataloader, device=device)
    y_true = predictions["y_true"]
    y_prob = predictions["y_prob"]
    metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=threshold)
    params = count_model_parameters(model)

    result: dict[str, Any] = {
        "model_name": model_name,
        **metrics.as_dict(),
        **params,
        "inference_samples": int(predictions["samples"]),
        "inference_elapsed_seconds": float(predictions["elapsed_seconds"]),
        "seconds_per_image": float(predictions["seconds_per_image"]),
    }

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        result["checkpoint_path"] = str(checkpoint_path)
        result["checkpoint_size_bytes"] = Path(checkpoint_path).stat().st_size

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame([result]).to_csv(output_dir / f"{model_name}_metrics.csv", index=False)
        save_json(result, output_dir / f"{model_name}_metrics.json")
        save_confusion_matrix_figure(result, output_dir / f"{model_name}_confusion_matrix.png")
        save_roc_curve_figure(
            y_true=y_true,
            y_prob=y_prob,
            model_name=model_name,
            output_path=output_dir / f"{model_name}_roc_curve.png",
        )

    return result


def save_confusion_matrix_figure(metrics: dict[str, Any], output_path: str | Path) -> Path:
    """Save a confusion matrix figure from computed metrics."""
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title(f"Matriz de confusao - {metrics['model_name']}")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_xticks([0, 1], labels=config.CLASS_NAMES)
    ax.set_yticks([0, 1], labels=config.CLASS_NAMES)

    for row in range(2):
        for col in range(2):
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center", color="black")

    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_roc_curve_figure(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    output_path: str | Path,
) -> Path | None:
    """Save ROC curve figure when both classes are present."""
    roc_data = compute_roc_curve(y_true, y_prob)
    if roc_data is None:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(roc_data["fpr"], roc_data["tpr"], label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aleatorio")
    ax.set_title(f"Curva ROC - {model_name}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path
