"""Image preprocessing and transform utilities for CardioIA Vision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from torchvision import transforms

from src import config


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TransformConfig:
    """Configuration for image preprocessing and augmentation."""

    image_size: int = config.IMAGE_SIZE
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    rotation_degrees: float = 10.0
    translate_fraction: tuple[float, float] = (0.03, 0.03)
    brightness: float = 0.10
    contrast: float = 0.10
    use_horizontal_flip: bool = False
    horizontal_flip_probability: float = 0.5


def load_image_rgb(image_path: str) -> Image.Image:
    """Load an image from disk and convert it to RGB."""
    return Image.open(image_path).convert("RGB")


def get_train_transforms(transform_config: TransformConfig | None = None) -> transforms.Compose:
    """Return training transforms with light medical-image augmentations."""
    cfg = transform_config or TransformConfig()

    augmentation_steps: list[transforms.Transform] = [
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.RandomAffine(
            degrees=cfg.rotation_degrees,
            translate=cfg.translate_fraction,
            fill=0,
        ),
        transforms.ColorJitter(brightness=cfg.brightness, contrast=cfg.contrast),
    ]

    # Disabled by default because horizontal flips can alter anatomical laterality.
    if cfg.use_horizontal_flip:
        augmentation_steps.append(
            transforms.RandomHorizontalFlip(p=cfg.horizontal_flip_probability)
        )

    augmentation_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )

    return transforms.Compose(augmentation_steps)


def get_eval_transforms(transform_config: TransformConfig | None = None) -> transforms.Compose:
    """Return deterministic transforms for validation, test, and inference."""
    cfg = transform_config or TransformConfig()
    return transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )


def get_transforms(
    split: str,
    transform_config: TransformConfig | None = None,
) -> transforms.Compose:
    """Return transforms for a named split."""
    normalized_split = split.lower()
    if normalized_split == "train":
        return get_train_transforms(transform_config)
    if normalized_split in {"val", "valid", "validation", "test", "infer", "inference"}:
        return get_eval_transforms(transform_config)
    raise ValueError(f"Unknown split for transforms: {split}")


def denormalize_tensor(
    image_tensor: torch.Tensor,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    """Undo ImageNet normalization for visualization."""
    if image_tensor.ndim != 3:
        raise ValueError("Expected image tensor with shape [C, H, W].")

    mean_tensor = torch.tensor(mean, dtype=image_tensor.dtype, device=image_tensor.device).view(
        -1, 1, 1
    )
    std_tensor = torch.tensor(std, dtype=image_tensor.dtype, device=image_tensor.device).view(
        -1, 1, 1
    )
    return (image_tensor * std_tensor + mean_tensor).clamp(0, 1)


def tensor_to_numpy_image(image_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a normalized CHW tensor to an HWC tensor ready for plotting."""
    return denormalize_tensor(image_tensor).permute(1, 2, 0).cpu()
