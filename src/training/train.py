"""Reusable training loop for CardioIA Vision models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src import config
from src.training.evaluate import collect_predictions, count_model_parameters
from src.training.experiment_logger import save_history, save_json
from src.training.metrics import compute_binary_metrics


@dataclass(frozen=True)
class TrainConfig:
    """Training runtime settings."""

    model_name: str
    epochs: int = config.DEFAULT_EPOCHS
    device: str = config.DEVICE
    threshold: float = config.DEFAULT_THRESHOLD
    metric_to_maximize: str = config.MODEL_METRIC_TO_MAXIMIZE
    checkpoint_dir: Path = config.CHECKPOINTS_DIR
    metrics_dir: Path = config.METRICS_DIR
    use_amp: bool = True
    gradient_clip_norm: float | None = None


def _prepare_targets(labels: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    """Prepare labels for BCE or CrossEntropy style outputs."""
    if outputs.ndim == 2 and outputs.shape[1] == 2:
        return labels.long()
    if outputs.ndim == 2 and outputs.shape[1] == 1:
        return labels.float().view_as(outputs)
    return labels.float()


def _batch_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(images)
    targets = _prepare_targets(labels, outputs)
    loss = criterion(outputs, targets)
    return loss, outputs


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    gradient_clip_norm: float | None = None,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    running_loss = 0.0
    total_samples = 0
    use_amp = scaler is not None and str(device).startswith("cuda")

    for batch in dataloader:
        if isinstance(batch, dict):
            images = batch["image"]
            labels = batch["label"]
        else:
            images, labels = batch

        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                loss, _ = _batch_loss(model, images, labels, criterion)
            scaler.scale(loss).backward()
            if gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, _ = _batch_loss(model, images, labels, criterion)
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()

        batch_size = int(labels.numel())
        running_loss += float(loss.detach().cpu().item()) * batch_size
        total_samples += batch_size

    return running_loss / total_samples if total_samples else 0.0


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str | torch.device,
) -> float:
    """Compute average loss for a dataloader."""
    model.eval()
    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        if isinstance(batch, dict):
            images = batch["image"]
            labels = batch["label"]
        else:
            images, labels = batch

        images = images.to(device)
        labels = labels.to(device)
        loss, _ = _batch_loss(model, images, labels, criterion)

        batch_size = int(labels.numel())
        running_loss += float(loss.detach().cpu().item()) * batch_size
        total_samples += batch_size

    return running_loss / total_samples if total_samples else 0.0


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric_value: float,
    train_config: TrainConfig,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a model checkpoint."""
    train_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = train_config.checkpoint_dir / f"{train_config.model_name}_best.pt"
    payload = {
        "model_name": train_config.model_name,
        "epoch": epoch,
        "metric_to_maximize": train_config.metric_to_maximize,
        "metric_value": metric_value,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "epochs": train_config.epochs,
            "threshold": train_config.threshold,
            "device": str(train_config.device),
        },
        "extra": extra or {},
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_config: TrainConfig,
    scheduler: Any | None = None,
) -> dict[str, Any]:
    """Train a model, save best checkpoint, and persist history."""
    device = torch.device(train_config.device)
    model.to(device)

    scaler = (
        torch.cuda.amp.GradScaler()
        if train_config.use_amp and str(device).startswith("cuda")
        else None
    )

    history: list[dict[str, Any]] = []
    best_metric = float("-inf")
    best_checkpoint_path: Path | None = None
    training_start = time.perf_counter()
    params = count_model_parameters(model)

    for epoch in range(1, train_config.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            gradient_clip_norm=train_config.gradient_clip_norm,
        )
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        val_predictions = collect_predictions(model, val_loader, device=device)
        val_metrics = compute_binary_metrics(
            y_true=val_predictions["y_true"],
            y_prob=val_predictions["y_prob"],
            threshold=train_config.threshold,
        )

        if scheduler is not None:
            scheduler.step()

        epoch_seconds = time.perf_counter() - epoch_start
        metric_value_raw = val_metrics.as_dict().get(train_config.metric_to_maximize)
        metric_value = float(metric_value_raw) if metric_value_raw is not None else float("-inf")

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{key}": value for key, value in val_metrics.as_dict().items()},
            "epoch_seconds": epoch_seconds,
            "learning_rate": optimizer.param_groups[0].get("lr"),
        }
        history.append(row)

        if metric_value > best_metric:
            best_metric = metric_value
            best_checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metric_value=metric_value,
                train_config=train_config,
                extra=params,
            )

    total_training_seconds = time.perf_counter() - training_start
    train_config.metrics_dir.mkdir(parents=True, exist_ok=True)
    history_path = train_config.metrics_dir / f"{train_config.model_name}_history.csv"
    summary_path = train_config.metrics_dir / f"{train_config.model_name}_training_summary.json"

    save_history(history, history_path)
    summary = {
        "model_name": train_config.model_name,
        "epochs": train_config.epochs,
        "best_metric": best_metric,
        "metric_to_maximize": train_config.metric_to_maximize,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path else None,
        "history_path": str(history_path),
        "total_training_seconds": total_training_seconds,
        "seconds_per_epoch": total_training_seconds / train_config.epochs
        if train_config.epochs
        else 0,
        **params,
    }
    save_json(summary, summary_path)

    return {
        "history": history,
        "summary": summary,
        "history_path": history_path,
        "summary_path": summary_path,
        "best_checkpoint_path": best_checkpoint_path,
    }
