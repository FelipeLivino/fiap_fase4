"""Inference helpers for the CardioIA Vision prototypes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn

from src import config
from src.models.custom_cnn import create_custom_cnn
from src.models.standard_cnn import create_standard_cnn
from src.models.transfer_learning import create_transfer_model
from src.models.vision_transformer import create_vision_transformer
from src.preprocessing import get_eval_transforms
from src.training.evaluate import _logits_to_probabilities


@dataclass(frozen=True)
class PredictionResult:
    """Structured prediction output for prototypes."""

    model_name: str
    predicted_label: str
    predicted_class: int
    probability_cardiomegaly: float
    probability_no_finding: float
    image_path: str


def find_exported_checkpoint(exported_dir: Path | None = None) -> Path:
    """Find the exported final checkpoint."""
    exported_dir = exported_dir or config.EXPORTED_MODELS_DIR
    candidates = sorted(exported_dir.glob("*_final.pt"))
    if not candidates:
        candidates = sorted(exported_dir.glob("*.pt"))

    if not candidates:
        raise FileNotFoundError(
            f"No exported model checkpoint found in {exported_dir}. "
            "Run notebooks/09_comparacao_modelos.ipynb after training models."
        )

    return candidates[0]


def infer_architecture_from_model_name(model_name: str) -> str:
    """Infer architecture key from a saved model name."""
    normalized = model_name.lower()
    if "cnn_propria" in normalized or "custom" in normalized:
        return "custom_cnn"
    if "cnn_padrao" in normalized or "baseline" in normalized or "standard" in normalized:
        return "standard_cnn"
    if "resnet50" in normalized:
        return "resnet50"
    if "efficientnet_b0" in normalized:
        return "efficientnet_b0"
    if "efficientnet_b3" in normalized:
        return "efficientnet_b3"
    if "densenet121" in normalized:
        return "densenet121"
    if "vit_b_16" in normalized:
        return "vit_b_16"
    if "swin_t" in normalized:
        return "swin_t"
    raise ValueError(
        f"Could not infer architecture from model_name={model_name!r}. "
        "Update src/inference.py with this model mapping."
    )


def create_model_for_architecture(architecture: str) -> nn.Module:
    """Create an untrained model matching an exported checkpoint architecture."""
    if architecture == "custom_cnn":
        return create_custom_cnn()
    if architecture == "standard_cnn":
        return create_standard_cnn()
    if architecture in {"resnet50", "efficientnet_b0", "efficientnet_b3", "densenet121"}:
        return create_transfer_model(
            architecture,
            pretrained=False,
            freeze_backbone=False,
        )
    if architecture in {"vit_b_16", "swin_t"}:
        return create_vision_transformer(
            architecture,
            pretrained=False,
            freeze_backbone=False,
        )
    raise ValueError(f"Unsupported architecture: {architecture}")


def load_model_from_checkpoint(
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = config.DEVICE,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load exported checkpoint and return model plus checkpoint metadata."""
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else find_exported_checkpoint()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", checkpoint_path.stem)
    architecture = infer_architecture_from_model_name(model_name)
    model = create_model_for_architecture(architecture)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "model_name": model_name,
        "architecture": architecture,
        "epoch": checkpoint.get("epoch"),
        "metric_to_maximize": checkpoint.get("metric_to_maximize"),
        "metric_value": checkpoint.get("metric_value"),
    }
    return model, metadata


@torch.no_grad()
def predict_image(
    image_path: str | Path,
    model: nn.Module,
    model_name: str,
    device: str | torch.device = config.DEVICE,
) -> PredictionResult:
    """Run inference for one image path."""
    image_path = Path(image_path)
    transform = get_eval_transforms()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probability_cardiomegaly = float(_logits_to_probabilities(logits).detach().cpu().item())
    predicted_class = int(probability_cardiomegaly >= config.DEFAULT_THRESHOLD)
    predicted_label = config.INDEX_TO_CLASS[predicted_class]

    return PredictionResult(
        model_name=model_name,
        predicted_label=predicted_label,
        predicted_class=predicted_class,
        probability_cardiomegaly=probability_cardiomegaly,
        probability_no_finding=1.0 - probability_cardiomegaly,
        image_path=str(image_path),
    )
