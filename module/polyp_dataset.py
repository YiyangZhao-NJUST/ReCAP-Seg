"""Polyp segmentation dataset for ReCAP-Seg.

The dataset returns image--mask pairs and optional structured attribute labels.
It is designed for a common polyp segmentation folder layout but does not depend
on any private paths.

Expected sample output:
    {
        "image": Tensor[3, H, W],
        "label": Tensor[1, H, W],
        "labels": Optional[Dict[str, List[str]]],
        "filename": str,
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset


LOGGER = logging.getLogger("ReCAP-Seg")


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class PolypDataset(Dataset):
    """Dataset for 2D polyp image segmentation with optional attribute labels."""

    def __init__(
        self,
        image_root: str | Path,
        mask_root: str | Path,
        label_file: Optional[str | Path] = None,
        image_size: Tuple[int, int] = (256, 256),
        augment: bool = True,
        split: str = "train",
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        """Initialize the dataset.

        Args:
            image_root: Directory containing input images.
            mask_root: Directory containing binary segmentation masks.
            label_file: Optional JSON file containing structured attribute labels.
            image_size: Target image size as (H, W).
            augment: Whether to apply training augmentation.
            split: Dataset split name, e.g., "train", "val", or "test".
            mean: Normalization mean.
            std: Normalization standard deviation.
        """
        super().__init__()

        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)
        self.label_file = Path(label_file) if label_file is not None else None
        self.image_size = tuple(image_size)
        self.augment = bool(augment) and split == "train"
        self.split = split
        self.mean = mean
        self.std = std

        if not self.image_root.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_root}")
        if not self.mask_root.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_root}")

        self.samples = self._load_samples()
        if not self.samples:
            raise RuntimeError(
                f"No valid image-mask pairs found under image_root={self.image_root} "
                f"and mask_root={self.mask_root}."
            )

        self.labels_dict: Dict[str, Dict[str, List[str]]] = {}
        if self.label_file is not None:
            if not self.label_file.is_file():
                raise FileNotFoundError(f"Label file not found: {self.label_file}")
            self.labels_dict = self._load_labels(self.label_file)
            LOGGER.info("Loaded attribute labels for %d samples", len(self.labels_dict))

        self.transform = self._create_transform()

        LOGGER.info("Loaded %d %s samples", len(self.samples), split)
        LOGGER.info("  Image root: %s", self.image_root)
        LOGGER.info("  Mask root : %s", self.mask_root)

    def _load_samples(self) -> List[str]:
        """Load sample stems that have both image and mask files."""
        samples: List[str] = []

        for image_path in sorted(self.image_root.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            mask_path = self._find_mask_path(image_path.stem)
            if mask_path is None:
                LOGGER.warning("No mask found for image: %s", image_path.name)
                continue

            samples.append(image_path.stem)

        return samples

    def _load_labels(self, label_file: Path) -> Dict[str, Dict[str, List[str]]]:
        """Load structured attribute labels from JSON.

        Supported formats:
            1. List format:
               [
                 {"filename": "xxx.png", "labels": {"shape": ["round"]}},
                 ...
               ]

            2. Dictionary format:
               {
                 "xxx.png": {"shape": "round", "boundary": ["clear"]},
                 ...
               }
        """
        with label_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        labels_dict: Dict[str, Dict[str, List[str]]] = {}

        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                filename = item.get("filename", item.get("name", ""))
                if not filename:
                    continue
                stem = Path(filename).stem
                labels = item.get("labels", item.get("attributes", {}))
                labels_dict[stem] = self._normalize_labels(labels)

        elif isinstance(data, dict):
            for filename, labels in data.items():
                stem = Path(filename).stem
                if isinstance(labels, dict) and "labels" in labels:
                    labels = labels["labels"]
                labels_dict[stem] = self._normalize_labels(labels)

        else:
            raise ValueError(f"Unsupported label file format: {label_file}")

        return labels_dict

    @staticmethod
    def _normalize_labels(labels: Any) -> Dict[str, List[str]]:
        """Normalize labels so each slot maps to a list of string values."""
        if labels is None:
            return {}
        if not isinstance(labels, dict):
            raise ValueError(f"Labels must be a dictionary, got {type(labels)}")

        normalized: Dict[str, List[str]] = {}
        invalid_slots = set(labels.get("invalid_slots", [])) if isinstance(labels.get("invalid_slots", []), list) else set()

        for slot, value in labels.items():
            if slot == "invalid_slots":
                continue
            if slot in invalid_slots:
                continue
            if value is None:
                continue
            if isinstance(value, list):
                values = [str(v) for v in value if v is not None]
            else:
                values = [str(value)]
            if values:
                normalized[slot] = values

        return normalized

    def _create_transform(self) -> A.Compose:
        """Create image and mask transformations."""
        height, width = self.image_size

        if self.augment:
            return A.Compose(
                [
                    A.LongestMaxSize(max_size=max(height, width)),
                    A.PadIfNeeded(
                        min_height=height,
                        min_width=width,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        mask_value=0,
                    ),
                    A.RandomResizedCrop(
                        size=(height, width),
                        scale=(0.85, 1.0),
                        ratio=(0.95, 1.05),
                        p=0.3,
                    ),
                    A.ShiftScaleRotate(
                        shift_limit=0.04,
                        scale_limit=0.10,
                        rotate_limit=8,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        mask_value=0,
                        p=0.5,
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
                            A.RandomGamma(gamma_limit=(85, 115), p=1.0),
                            A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=1.0),
                        ],
                        p=0.5,
                    ),
                    A.OneOf(
                        [
                            A.GaussNoise(var_limit=(5.0, 15.0), p=1.0),
                            A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=1.0),
                        ],
                        p=0.25,
                    ),
                    A.OneOf(
                        [
                            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                            A.MotionBlur(blur_limit=3, p=1.0),
                            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.9, 1.1), p=1.0),
                        ],
                        p=0.2,
                    ),
                    A.Normalize(mean=self.mean, std=self.std),
                    ToTensorV2(),
                ]
            )

        return A.Compose(
            [
                A.LongestMaxSize(max_size=max(height, width)),
                A.PadIfNeeded(
                    min_height=height,
                    min_width=width,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                ),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ]
        )

    def _find_image_path(self, filename: str) -> Optional[Path]:
        """Find an image file by stem."""
        return self._find_file(self.image_root, filename)

    def _find_mask_path(self, filename: str) -> Optional[Path]:
        """Find a mask file by stem."""
        return self._find_file(self.mask_root, filename)

    @staticmethod
    def _find_file(root: Path, filename: str) -> Optional[Path]:
        """Find a file with one of the supported image extensions."""
        for ext in IMAGE_EXTENSIONS:
            path = root / f"{filename}{ext}"
            if path.is_file():
                return path
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        filename = self.samples[index]

        image_path = self._find_image_path(filename)
        mask_path = self._find_mask_path(filename)
        if image_path is None:
            raise FileNotFoundError(f"Image file not found for sample: {filename}")
        if mask_path is None:
            raise FileNotFoundError(f"Mask file not found for sample: {filename}")

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image-mask size mismatch for {filename}: "
                f"image={image.shape[:2]}, mask={mask.shape[:2]}"
            )

        mask = (mask > 127).astype(np.float32)
        transformed = self.transform(image=image, mask=mask)

        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"].float().unsqueeze(0)

        labels = self.labels_dict.get(filename) if self.labels_dict else None

        return {
            "image": image_tensor,
            "label": mask_tensor,
            "labels": labels,
            "filename": filename,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate samples with optional variable-length structured attributes."""
    images = torch.stack([item["image"] for item in batch], dim=0)
    masks = torch.stack([item["label"] for item in batch], dim=0)
    labels = [item.get("labels") for item in batch]
    filenames = [item.get("filename", "") for item in batch]

    return {
        "image": images,
        "label": masks,
        "labels": labels,
        "filename": filenames,
    }


def create_dataloaders(
    train_image_root: str | Path,
    train_mask_root: str | Path,
    val_image_root: str | Path,
    val_mask_root: str | Path,
    train_label_file: Optional[str | Path] = None,
    val_label_file: Optional[str | Path] = None,
    image_size: Tuple[int, int] = (256, 256),
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    train_dataset = PolypDataset(
        image_root=train_image_root,
        mask_root=train_mask_root,
        label_file=train_label_file,
        image_size=image_size,
        augment=True,
        split="train",
    )

    val_dataset = PolypDataset(
        image_root=val_image_root,
        mask_root=val_mask_root,
        label_file=val_label_file,
        image_size=image_size,
        augment=False,
        split="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
