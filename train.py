#!/usr/bin/env python3
"""Training entry point for ReCAP-Seg.

Examples:
    python train.py --config configs/polyp.yaml
    python train.py --config configs/polyp.yaml --resume checkpoints/recapseg_polyp/latest.pth
    CUDA_VISIBLE_DEVICES=0 python train.py --config configs/polyp.yaml

The script expects a YAML config file containing DATA, MODEL, TRAIN, and VAL fields.
Command-line arguments can override common training options such as batch size,
learning rate, number of epochs, random seed, and data root.
"""

from __future__ import annotations

import argparse
import copy
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


# Make local project modules importable when running `python train.py`.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------
# Recommended public names for the GitHub release. If your current code still
# uses the old internal names, keep the fallback imports until you rename files.
from models import ReCAPSeg
from modules.trainer import ReCAPSegTrainer
from modules.polyp_dataset import PolypDataset



LOGGER = logging.getLogger("ReCAP-Seg")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train ReCAP-Seg")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/polyp.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint for resuming training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device type. Use CUDA_VISIBLE_DEVICES to select visible GPUs.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Optional local GPU index. If omitted, PyTorch uses the default CUDA device.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Overrides SEED in the config.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size. Overrides TRAIN.batch_size in the config.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs. Overrides TRAIN.epochs in the config.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Base learning rate. Overrides TRAIN.base_lr in the config.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Dataset root directory. Overrides DATA.data_root in the config.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of DataLoader workers. Overrides NUM_WORKERS in the config.",
    )
    parser.add_argument(
        "--non_deterministic",
        action="store_true",
        help="Disable deterministic cuDNN behavior for potentially faster training.",
    )

    return parser.parse_args()


def setup_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {config_path}")

    return config


def update_config_from_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply command-line overrides to the loaded config."""
    config = copy.deepcopy(config)
    config.setdefault("DATA", {})
    config.setdefault("TRAIN", {})
    config.setdefault("VAL", {})

    if args.seed is not None:
        config["SEED"] = args.seed
    if args.batch_size is not None:
        config["TRAIN"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["TRAIN"]["epochs"] = args.epochs
    if args.lr is not None:
        config["TRAIN"]["base_lr"] = args.lr
    if args.data_root is not None:
        config["DATA"]["data_root"] = args.data_root
    if args.num_workers is not None:
        config["NUM_WORKERS"] = args.num_workers

    return config


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    LOGGER.info("Random seed set to %d", seed)


def setup_device(device_type: str = "cuda", gpu: Optional[int] = None) -> torch.device:
    """Create the torch device used for training."""
    if device_type == "cuda" and torch.cuda.is_available():
        if gpu is not None:
            torch.cuda.set_device(gpu)
            device = torch.device(f"cuda:{gpu}")
        else:
            device = torch.device("cuda")
        LOGGER.info("Using CUDA device: %s", torch.cuda.get_device_name(device))
        return device

    if device_type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA was requested but is not available. Falling back to CPU.")

    return torch.device("cpu")


def _resolve_path(path: Optional[str], root: Path) -> Optional[str]:
    """Resolve a path relative to the project root when needed."""
    if path is None:
        return None
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    return str(root / path_obj)


def _validate_path(path: Optional[str], name: str, required: bool = True) -> None:
    """Check whether a configured path exists."""
    if path is None:
        if required:
            raise ValueError(f"Missing required path: {name}")
        return

    if required and not Path(path).exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")


def default_split_paths(data_root: str | Path) -> Dict[str, str]:
    """Return default paths for the commonly used polyp split.

    Users can override these paths in the YAML config with explicit fields:
    DATA.train_image_root, DATA.train_mask_root, DATA.val_image_root, and
    DATA.val_mask_root.
    """
    data_root = Path(data_root)
    return {
        "train_image_root": str(data_root / "TrainDataset" / "images"),
        "train_mask_root": str(data_root / "TrainDataset" / "masks"),
        "val_image_root": str(data_root / "TestDataset" / "Kvasir" / "images"),
        "val_mask_root": str(data_root / "TestDataset" / "Kvasir" / "masks"),
    }


def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    data_cfg = config.get("DATA", {})
    train_cfg = config.get("TRAIN", {})
    val_cfg = config.get("VAL", {})

    image_size = tuple(data_cfg.get("image_size", [256, 256]))
    data_root = data_cfg.get("data_root", "data/polyp")

    paths = default_split_paths(data_root)
    paths.update({k: v for k, v in data_cfg.items() if k in paths and v is not None})

    train_image_root = _resolve_path(paths["train_image_root"], PROJECT_ROOT)
    train_mask_root = _resolve_path(paths["train_mask_root"], PROJECT_ROOT)
    val_image_root = _resolve_path(paths["val_image_root"], PROJECT_ROOT)
    val_mask_root = _resolve_path(paths["val_mask_root"], PROJECT_ROOT)
    train_label_file = _resolve_path(data_cfg.get("train_label_file"), PROJECT_ROOT)
    val_label_file = _resolve_path(data_cfg.get("val_label_file"), PROJECT_ROOT)

    _validate_path(train_image_root, "DATA.train_image_root")
    _validate_path(train_mask_root, "DATA.train_mask_root")
    _validate_path(val_image_root, "DATA.val_image_root")
    _validate_path(val_mask_root, "DATA.val_mask_root")
    _validate_path(train_label_file, "DATA.train_label_file", required=False)
    _validate_path(val_label_file, "DATA.val_label_file", required=False)

    train_batch_size = int(train_cfg.get("batch_size", 8))
    val_batch_size = int(val_cfg.get("batch_size", train_batch_size))
    num_workers = int(config.get("NUM_WORKERS", 4))
    pin_memory = bool(config.get("PIN_MEMORY", True))

    LOGGER.info("Train image root: %s", train_image_root)
    LOGGER.info("Train mask root : %s", train_mask_root)
    LOGGER.info("Val image root  : %s", val_image_root)
    LOGGER.info("Val mask root   : %s", val_mask_root)
    LOGGER.info("Image size      : %s", image_size)
    LOGGER.info("Train batch size: %d", train_batch_size)
    LOGGER.info("Val batch size  : %d", val_batch_size)

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
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )

    LOGGER.info("Train samples: %d", len(train_dataset))
    LOGGER.info("Val samples  : %d", len(val_dataset))

    return train_loader, val_loader


def collate_fn(batch: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a batch returned by the dataset.

    Expected item fields:
        image: Tensor[C, H, W]
        label: Tensor[1, H, W] or Tensor[H, W]
        labels: optional structured attribute labels
        filename: optional image file name
    """
    batch = list(batch)
    images = torch.stack([item["image"] for item in batch], dim=0)
    masks = torch.stack([item["label"] for item in batch], dim=0)
    attribute_labels = [item.get("labels") for item in batch]
    filenames = [item.get("filename", "") for item in batch]

    return {
        "image": images,
        "label": masks,
        "labels": attribute_labels,
        "filename": filenames,
    }


class ModelArgs:
    """Minimal argument container for model constructors requiring args.device."""

    def __init__(self, device: torch.device) -> None:
        self.device = str(device)


def create_model(config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Build the ReCAP-Seg model from config."""
    model_cfg = config.get("MODEL", {})
    encoder_cfg = model_cfg.get("encoder", {})
    prototype_cfg = model_cfg.get("prototype", {})
    decoder_cfg = model_cfg.get("decoder", {})

    model = ReCAPSeg(
        args=ModelArgs(device),
        encoder_name=encoder_cfg.get("name", "convnextv2_base"),
        embed_dim=int(model_cfg.get("embed_dim", 512)),
        frozen_stages=int(encoder_cfg.get("frozen_stages", 0)),
        num_heads=int(decoder_cfg.get("num_heads", 8)),
        use_text_init=bool(prototype_cfg.get("use_text_init", True)),
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    LOGGER.info("Model created")
    LOGGER.info("  Total params    : %.2fM", total_params / 1e6)
    LOGGER.info("  Trainable params: %.2fM", trainable_params / 1e6)
    LOGGER.info("  Encoder         : %s", encoder_cfg.get("name", "convnextv2_base"))
    LOGGER.info("  Frozen stages   : %d", int(encoder_cfg.get("frozen_stages", 0)))

    return model


def main() -> None:
    setup_logging()
    args = parse_args()

    config = load_config(args.config)
    config = update_config_from_args(config, args)

    device = setup_device(args.device, args.gpu)
    seed = int(config.get("SEED", 42))
    set_seed(seed, deterministic=not args.non_deterministic)

    train_loader, val_loader = create_dataloaders(config)
    model = create_model(config, device)

    trainer = ReCAPSegTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        LOGGER.info("Resuming training from: %s", resume_path)
        trainer.load_checkpoint(str(resume_path))

    LOGGER.info("%s", "=" * 80)
    LOGGER.info("Starting ReCAP-Seg training")
    LOGGER.info("Start time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    LOGGER.info("%s", "=" * 80)

    trainer.train()

    LOGGER.info("Training completed")
    if hasattr(trainer, "save_dir"):
        LOGGER.info("Best checkpoint: %s", Path(trainer.save_dir) / "best.pth")


if __name__ == "__main__":
    main()
