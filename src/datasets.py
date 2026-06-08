"""PyTorch datasets and dataloaders for the CardioIA Vision binary task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src import config
from src.preprocessing import get_transforms, load_image_rgb


REQUIRED_SPLIT_COLUMNS = {
    "Image Index",
    "Patient ID",
    "Finding Labels",
    "binary_label",
    "binary_label_name",
}


@dataclass(frozen=True)
class DataLoaderConfig:
    """Runtime settings for DataLoader creation."""

    batch_size: int = config.BATCH_SIZE
    num_workers: int = config.NUM_WORKERS
    pin_memory: bool = True
    persistent_workers: bool = False
    drop_last_train: bool = False
    use_weighted_sampler: bool = False


class ChestXrayBinaryDataset(Dataset):
    """Dataset for binary NIH Chest X-ray classification.

    The split CSV must contain a binary target in `binary_label`, where:
    0 = No Finding and 1 = Cardiomegaly.
    """

    def __init__(
        self,
        split_csv: str | Path,
        transform: Any | None = None,
        images_dir: str | Path | None = None,
        project_root: str | Path | None = None,
        return_metadata: bool = False,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.images_dir = Path(images_dir) if images_dir is not None else config.RAW_IMAGES_DIR
        self.project_root = Path(project_root) if project_root is not None else config.PROJECT_ROOT
        self.transform = transform
        self.return_metadata = return_metadata

        if not self.split_csv.exists():
            raise FileNotFoundError(
                f"Split CSV not found: {self.split_csv}. "
                "Run notebooks/02_preprocessamento_splits.ipynb first."
            )

        self.data = pd.read_csv(self.split_csv)
        missing_columns = REQUIRED_SPLIT_COLUMNS - set(self.data.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {self.split_csv}: {sorted(missing_columns)}"
            )

        self.data["binary_label"] = self.data["binary_label"].astype(int)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor] | dict[str, Any]:
        row = self.data.iloc[index]
        image_path = self.resolve_image_path(row)
        image = load_image_rgb(str(image_path))

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(row["binary_label"], dtype=torch.float32)

        if not self.return_metadata:
            return image, label

        return {
            "image": image,
            "label": label,
            "image_index": row["Image Index"],
            "patient_id": row["Patient ID"],
            "finding_labels": row["Finding Labels"],
            "binary_label_name": row["binary_label_name"],
            "image_path": str(image_path),
        }

    def resolve_image_path(self, row: pd.Series) -> Path:
        """Resolve an image path from absolute_path, relative_path, or Image Index."""
        absolute_path = row.get("absolute_path")
        if isinstance(absolute_path, str) and absolute_path and absolute_path != "nan":
            path = Path(absolute_path)
            if path.exists():
                return path

        relative_path = row.get("relative_path")
        if isinstance(relative_path, str) and relative_path and relative_path != "nan":
            path = self.project_root / relative_path
            if path.exists():
                return path

        fallback_path = self.images_dir / str(row["Image Index"])
        if fallback_path.exists():
            return fallback_path

        raise FileNotFoundError(
            f"Image file not found for {row['Image Index']}. Checked absolute_path, "
            f"relative_path, and fallback path {fallback_path}."
        )

    @property
    def class_counts(self) -> dict[int, int]:
        return {
            int(label): int(count)
            for label, count in self.data["binary_label"].value_counts().sort_index().items()
        }

    @property
    def class_weights(self) -> torch.Tensor:
        counts = self.data["binary_label"].value_counts().sort_index()
        weights = len(self.data) / (len(counts) * counts)
        return torch.tensor(weights.to_numpy(), dtype=torch.float32)


def create_weighted_sampler(dataset: ChestXrayBinaryDataset) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler from inverse class frequencies."""
    counts = dataset.data["binary_label"].value_counts().to_dict()
    sample_weights = dataset.data["binary_label"].map(lambda label: 1.0 / counts[label])
    weights_tensor = torch.tensor(sample_weights.to_numpy(), dtype=torch.double)
    return WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(weights_tensor),
        replacement=True,
    )


def create_dataset(
    split_name: str,
    split_csv: str | Path | None = None,
    return_metadata: bool = False,
) -> ChestXrayBinaryDataset:
    """Create a dataset for train, val, or test using standard transforms."""
    if split_csv is None:
        split_csv = config.SPLITS_DIR / f"{split_name}.csv"

    transform = get_transforms(split_name)
    return ChestXrayBinaryDataset(
        split_csv=split_csv,
        transform=transform,
        return_metadata=return_metadata,
    )


def create_dataloader(
    dataset: ChestXrayBinaryDataset,
    split_name: str,
    dataloader_config: DataLoaderConfig | None = None,
) -> DataLoader:
    """Create a DataLoader for a dataset and split."""
    cfg = dataloader_config or DataLoaderConfig()
    is_train = split_name.lower() == "train"
    sampler = create_weighted_sampler(dataset) if is_train and cfg.use_weighted_sampler else None
    shuffle = is_train and sampler is None
    persistent_workers = cfg.persistent_workers and cfg.num_workers > 0

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=persistent_workers,
        drop_last=is_train and cfg.drop_last_train,
    )


def create_dataloaders(
    splits_dir: str | Path | None = None,
    dataloader_config: DataLoaderConfig | None = None,
    return_metadata: bool = False,
) -> dict[str, DataLoader]:
    """Create train, validation, and test DataLoaders from split CSVs."""
    splits_dir = Path(splits_dir) if splits_dir is not None else config.SPLITS_DIR
    loaders: dict[str, DataLoader] = {}

    for split_name in ["train", "val", "test"]:
        dataset = create_dataset(
            split_name=split_name,
            split_csv=splits_dir / f"{split_name}.csv",
            return_metadata=return_metadata,
        )
        loaders[split_name] = create_dataloader(
            dataset=dataset,
            split_name=split_name,
            dataloader_config=dataloader_config,
        )

    return loaders


def inspect_batch(batch: tuple[torch.Tensor, torch.Tensor] | dict[str, Any]) -> dict[str, Any]:
    """Return simple shape and label diagnostics for a DataLoader batch."""
    if isinstance(batch, dict):
        images = batch["image"]
        labels = batch["label"]
    else:
        images, labels = batch

    return {
        "images_shape": tuple(images.shape),
        "labels_shape": tuple(labels.shape),
        "images_dtype": str(images.dtype),
        "labels_dtype": str(labels.dtype),
        "label_values": sorted(labels.detach().cpu().unique().tolist()),
    }
