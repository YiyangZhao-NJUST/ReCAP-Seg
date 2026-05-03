#!/usr/bin/env python3
"""Testing and inference entry point for ReCAP-Seg.

Examples:
    python test.py --config configs/polyp.yaml --checkpoint checkpoints/recapseg_polyp/best.pth
    python test.py --config configs/polyp.yaml --checkpoint checkpoints/recapseg_polyp/best.pth --test_dataset CVC-ClinicDB
    CUDA_VISIBLE_DEVICES=0 python test.py --config configs/polyp.yaml --checkpoint checkpoints/recapseg_polyp/best.pth

This script supports:
    1. segmentation evaluation with Dice and mIoU;
    2. saving predicted masks;
    3. saving overlay visualizations;
    4. saving optional slot-wise attribute predictions to JSON.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


# Make local project modules importable when running `python test.py`.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------
# Keep these names consistent with the public GitHub release.
from models import ReCAPSeg
from modules.polyp_dataset import PolypDataset


LOGGER = logging.getLogger("ReCAP-Seg")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test ReCAP-Seg")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/polyp.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint. If omitted, LOG.save_dir/best.pth in the config is used.",
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
        "--data_root",
        type=str,
        default=None,
        help="Dataset root directory. Overrides DATA.data_root in the config.",
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default=None,
        help="Name of the test subset under DATA.data_root/TestDataset, e.g., Kvasir or CVC-ClinicDB.",
    )
    parser.add_argument(
        "--test_image_root",
        type=str,
        default=None,
        help="Explicit test image directory. Overrides config and --test_dataset.",
    )
    parser.add_argument(
        "--test_mask_root",
        type=str,
        default=None,
        help="Explicit test mask directory. Overrides config and --test_dataset.",
    )
    parser.add_argument(
        "--test_label_file",
        type=str,
        default=None,
        help="Optional structured attribute label file for test-time analysis.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/test",
        help="Directory for saving masks, overlays, and JSON results.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Testing batch size. Use 1 when saving per-image visualizations.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of DataLoader workers. Overrides NUM_WORKERS in the config.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for converting probabilities into binary masks.",
    )
    parser.add_argument(
        "--save_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save predicted segmentation masks.",
    )
    parser.add_argument(
        "--save_overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save overlay visualizations.",
    )
    parser.add_argument(
        "--save_attributes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save predicted slot-wise attributes when provided by the model.",
    )
    parser.add_argument(
        "--non_deterministic",
        action="store_true",
        help="Disable deterministic cuDNN behavior for potentially faster inference.",
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
    config.setdefault("TEST", {})

    if args.seed is not None:
        config["SEED"] = args.seed
    if args.data_root is not None:
        config["DATA"]["data_root"] = args.data_root
    if args.test_image_root is not None:
        config["DATA"]["test_image_root"] = args.test_image_root
    if args.test_mask_root is not None:
        config["DATA"]["test_mask_root"] = args.test_mask_root
    if args.test_label_file is not None:
        config["DATA"]["test_label_file"] = args.test_label_file
    if args.num_workers is not None:
        config["NUM_WORKERS"] = args.num_workers

    config["TEST"]["batch_size"] = args.batch_size
    config["TEST"]["threshold"] = args.threshold

    return config


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducible evaluation."""
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
    """Create the torch device used for testing."""
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


def default_test_paths(data_root: str | Path, test_dataset: str = "Kvasir") -> Dict[str, str]:
    """Return default test paths for a polyp-style TestDataset layout."""
    data_root = Path(data_root)
    test_root = data_root / "TestDataset" / test_dataset
    return {
        "test_image_root": str(test_root / "images"),
        "test_mask_root": str(test_root / "masks"),
    }


def create_test_loader(config: Dict[str, Any], args: argparse.Namespace) -> DataLoader:
    """Create the test dataloader."""
    data_cfg = config.get("DATA", {})
    test_cfg = config.get("TEST", {})

    image_size = tuple(data_cfg.get("image_size", [256, 256]))
    data_root = data_cfg.get("data_root", "data/polyp")
    test_dataset = args.test_dataset or data_cfg.get("test_dataset", "Kvasir")

    paths = default_test_paths(data_root, test_dataset)
    paths.update({k: v for k, v in data_cfg.items() if k in paths and v is not None})

    test_image_root = _resolve_path(paths["test_image_root"], PROJECT_ROOT)
    test_mask_root = _resolve_path(paths["test_mask_root"], PROJECT_ROOT)
    test_label_file = _resolve_path(data_cfg.get("test_label_file"), PROJECT_ROOT)

    _validate_path(test_image_root, "DATA.test_image_root")
    _validate_path(test_mask_root, "DATA.test_mask_root")
    _validate_path(test_label_file, "DATA.test_label_file", required=False)

    batch_size = int(test_cfg.get("batch_size", args.batch_size))
    num_workers = int(config.get("NUM_WORKERS", 4))
    pin_memory = bool(config.get("PIN_MEMORY", True))

    LOGGER.info("Test dataset    : %s", test_dataset)
    LOGGER.info("Test image root : %s", test_image_root)
    LOGGER.info("Test mask root  : %s", test_mask_root)
    LOGGER.info("Test label file : %s", test_label_file)
    LOGGER.info("Image size      : %s", image_size)
    LOGGER.info("Test batch size : %d", batch_size)

    test_dataset_obj = PolypDataset(
        image_root=test_image_root,
        mask_root=test_mask_root,
        label_file=test_label_file,
        image_size=image_size,
        augment=False,
        split="test",
    )

    test_loader = DataLoader(
        test_dataset_obj,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )

    LOGGER.info("Test samples    : %d", len(test_dataset_obj))
    return test_loader


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


def resolve_checkpoint_path(config: Dict[str, Any], checkpoint_arg: Optional[str]) -> Path:
    """Resolve checkpoint path from CLI or config."""
    if checkpoint_arg is not None:
        checkpoint_path = Path(checkpoint_arg)
    else:
        log_cfg = config.get("LOG", {})
        save_dir = Path(log_cfg.get("save_dir", "checkpoints/recapseg"))
        checkpoint_path = save_dir / "best.pth"

    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return checkpoint_path


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    """Load model weights from a checkpoint."""
    LOGGER.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Support checkpoints saved from DistributedDataParallel/DataParallel.
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        cleaned_state_dict[cleaned_key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    LOGGER.info("Checkpoint loaded successfully")


def compute_dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute batch-averaged Dice score."""
    pred = pred.float().flatten(1)
    target = target.float().flatten(1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


def compute_miou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute batch-averaged mIoU score."""
    pred = pred.float().flatten(1)
    target = target.float().flatten(1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def normalize_mask_shape(mask: torch.Tensor) -> torch.Tensor:
    """Ensure mask tensor has shape [B, 1, H, W]."""
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError(f"Expected mask shape [B, 1, H, W] or [B, H, W], got {tuple(mask.shape)}")
    return mask


def update_attribute_accuracy(
    pred_attrs: Dict[str, Dict[str, Any]],
    gt_labels: List[Optional[Dict[str, List[str]]]],
    correct: Dict[str, int],
    total: Dict[str, int],
) -> None:
    """Update slot-wise attribute accuracy when ground-truth attributes are available."""
    for slot, pred_info in pred_attrs.items():
        correct.setdefault(slot, 0)
        total.setdefault(slot, 0)

        pred_values = pred_info.get("pred_values", [])
        for idx, gt_label in enumerate(gt_labels):
            if gt_label is None or slot not in gt_label or idx >= len(pred_values):
                continue

            gt_values = gt_label[slot]
            if not gt_values or gt_values[0] in {"unknown", "not detected", "invalid"}:
                continue

            if pred_values[idx] in gt_values:
                correct[slot] += 1
            total[slot] += 1


def save_segmentation_mask(pred_mask: np.ndarray, save_path: str | Path) -> None:
    """Save a binary mask as an 8-bit PNG image."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    pred_mask = (pred_mask > 0).astype(np.uint8) * 255
    Image.fromarray(pred_mask, mode="L").save(save_path)


def denormalize_image(
    image: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Convert a normalized tensor image [3, H, W] to uint8 RGB image [H, W, 3]."""
    mean_tensor = torch.tensor(mean, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=image.dtype, device=image.device).view(3, 1, 1)

    image = image.detach() * std_tensor + mean_tensor
    image = image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (image * 255).astype(np.uint8)


def save_overlay_image(
    image: np.ndarray,
    pred_mask: np.ndarray,
    save_path: str | Path,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Save an RGB overlay visualization."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if image.shape[:2] != pred_mask.shape:
        resized_mask = Image.fromarray((pred_mask > 0).astype(np.uint8) * 255)
        resized_mask = resized_mask.resize((image.shape[1], image.shape[0]), Image.NEAREST)
        pred_mask = np.array(resized_mask) > 0
    else:
        pred_mask = pred_mask > 0

    overlay = image.astype(np.float32).copy()
    color_mask = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    mask_3d = pred_mask.astype(np.float32)[..., None]

    overlay = overlay * (1.0 - alpha * mask_3d) + color_mask * (alpha * mask_3d)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(save_path)


def safe_stem(filename: str, index: int) -> str:
    """Return a safe file stem for saving outputs."""
    if not filename:
        return f"sample_{index:05d}"
    return Path(filename).stem


def extract_attribute_predictions(
    pred_attrs: Dict[str, Dict[str, Any]],
    sample_index: int,
) -> Dict[str, Dict[str, Any]]:
    """Extract per-sample attribute predictions from batched model output."""
    attributes: Dict[str, Dict[str, Any]] = {}

    for slot, pred_info in pred_attrs.items():
        pred_values = pred_info.get("pred_values", [])
        pred_value = pred_values[sample_index] if sample_index < len(pred_values) else "unknown"

        probs_slot = pred_info.get("probs")
        if probs_slot is not None and sample_index < probs_slot.shape[0]:
            probs_list = probs_slot[sample_index].detach().cpu().numpy().tolist()
            confidence = float(max(probs_list)) if probs_list else 0.0
        else:
            probs_list = []
            confidence = 0.0

        attributes[slot] = {
            "predicted_value": pred_value,
            "confidence": confidence,
            "all_probs": probs_list,
        }

    return attributes


def get_attribute_slot_info(model: torch.nn.Module) -> Dict[str, Dict[str, Any]]:
    """Read attribute slot metadata from the model when available."""
    slot_info: Dict[str, Dict[str, Any]] = {}

    prototype_bank = None
    if hasattr(model, "prototype_lib"):
        prototype_bank = getattr(model, "prototype_lib")
    elif hasattr(model, "prototype_bank"):
        prototype_bank = getattr(model, "prototype_bank")

    if prototype_bank is None:
        return slot_info

    if not hasattr(prototype_bank, "get_slot_names"):
        return slot_info

    for slot in prototype_bank.get_slot_names():
        slot_entry: Dict[str, Any] = {}
        if hasattr(prototype_bank, "attribute_config") and slot in prototype_bank.attribute_config:
            config = prototype_bank.attribute_config[slot]
            slot_entry["values"] = config.get("values", [])
            slot_entry["weight"] = config.get("weight", 1.0)
        slot_info[slot] = slot_entry

    return slot_info


def run_inference(
    model: torch.nn.Module,
    images: torch.Tensor,
    threshold: float,
) -> Dict[str, Any]:
    """Run model inference with support for two-stage ReCAP-Seg inference."""
    if hasattr(model, "inference_two_stage"):
        return model.inference_two_stage(
            images,
            threshold=threshold,
            return_intermediate=True,
        )

    output = model(images)
    if isinstance(output, dict):
        return output
    return {"logits": output}


def evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    threshold: float,
    save_mask: bool,
    save_overlay: bool,
    save_attributes: bool,
    config_path: str,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    """Run testing, save outputs, and return a result summary."""
    model.eval()

    masks_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlay"
    if save_mask:
        masks_dir.mkdir(parents=True, exist_ok=True)
    if save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    dice_meter: List[float] = []
    miou_meter: List[float] = []
    attr_correct: Dict[str, int] = {}
    attr_total: Dict[str, int] = {}
    all_predictions: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing", ncols=100)):
            images = batch["image"].to(device, non_blocking=True)
            masks = normalize_mask_shape(batch["label"].to(device, non_blocking=True))
            filenames = batch.get("filename", [""] * images.shape[0])
            gt_labels = batch.get("labels", [None] * images.shape[0])

            output = run_inference(model, images, threshold=threshold)
            logits = output["logits"]
            logits = normalize_mask_shape(logits)

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            targets = (masks > 0.5).float()

            dice = compute_dice(preds, targets)
            miou = compute_miou(preds, targets)
            dice_meter.append(float(dice.item()))
            miou_meter.append(float(miou.item()))

            batch_size = images.shape[0]
            for sample_idx in range(batch_size):
                global_idx = batch_idx * batch_size + sample_idx
                filename = filenames[sample_idx]
                file_stem = safe_stem(filename, global_idx)
                pred_mask = preds[sample_idx, 0].detach().cpu().numpy()

                if save_mask:
                    save_segmentation_mask(pred_mask, masks_dir / f"{file_stem}.png")

                if save_overlay:
                    image_np = denormalize_image(images[sample_idx])
                    save_overlay_image(image_np, pred_mask, overlay_dir / f"{file_stem}_overlay.png")

                sample_result: Dict[str, Any] = {
                    "filename": filename,
                    "dice": float(dice.item()),
                    "miou": float(miou.item()),
                }

                if save_attributes and "pred_attributes" in output:
                    sample_result["predicted_attributes"] = extract_attribute_predictions(
                        output["pred_attributes"], sample_idx
                    )

                if sample_idx < len(gt_labels) and gt_labels[sample_idx] is not None:
                    sample_result["ground_truth_attributes"] = gt_labels[sample_idx]

                all_predictions.append(sample_result)

            if "pred_attributes" in output and gt_labels is not None:
                update_attribute_accuracy(output["pred_attributes"], gt_labels, attr_correct, attr_total)

    mean_dice = float(np.mean(dice_meter)) if dice_meter else 0.0
    mean_miou = float(np.mean(miou_meter)) if miou_meter else 0.0

    attr_accuracy = {}
    for slot, correct in attr_correct.items():
        total = attr_total.get(slot, 0)
        if total > 0:
            attr_accuracy[slot] = {
                "correct": correct,
                "total": total,
                "accuracy": correct / total,
            }

    results_summary: Dict[str, Any] = {
        "test_info": {
            "config": config_path,
            "checkpoint": str(checkpoint_path),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": len(all_predictions),
            "threshold": threshold,
        },
        "metrics": {
            "mean_dice": mean_dice,
            "mean_miou": mean_miou,
        },
        "attribute_accuracy": attr_accuracy,
        "attribute_slots": get_attribute_slot_info(model),
        "predictions": all_predictions,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.json"
    with predictions_path.open("w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    if save_attributes:
        simple_predictions = []
        for pred in all_predictions:
            simple_pred = {
                "filename": pred["filename"],
                "predicted_attributes": {},
            }
            for slot, info in pred.get("predicted_attributes", {}).items():
                simple_pred["predicted_attributes"][slot] = info.get("predicted_value", "unknown")
            if "ground_truth_attributes" in pred:
                simple_pred["ground_truth_attributes"] = pred["ground_truth_attributes"]
            simple_predictions.append(simple_pred)

        simple_path = output_dir / "attributes_simple.json"
        with simple_path.open("w", encoding="utf-8") as f:
            json.dump(simple_predictions, f, indent=2, ensure_ascii=False)

    LOGGER.info("Predictions saved to: %s", predictions_path)
    return results_summary


def main() -> None:
    setup_logging()
    args = parse_args()

    config = load_config(args.config)
    config = update_config_from_args(config, args)

    device = setup_device(args.device, args.gpu)
    seed = int(config.get("SEED", 42))
    set_seed(seed, deterministic=not args.non_deterministic)

    checkpoint_path = resolve_checkpoint_path(config, args.checkpoint)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"test_{timestamp}"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Output directory: %s", output_dir)

    test_loader = create_test_loader(config, args)
    model = create_model(config, device)
    load_checkpoint(model, checkpoint_path, device)

    LOGGER.info("%s", "=" * 80)
    LOGGER.info("Starting ReCAP-Seg testing")
    LOGGER.info("Start time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    LOGGER.info("%s", "=" * 80)

    results = evaluate(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        threshold=float(config.get("TEST", {}).get("threshold", args.threshold)),
        save_mask=args.save_mask,
        save_overlay=args.save_overlay,
        save_attributes=args.save_attributes,
        config_path=args.config,
        checkpoint_path=checkpoint_path,
    )

    LOGGER.info("%s", "=" * 80)
    LOGGER.info("Test results")
    LOGGER.info("Dice: %.4f", results["metrics"]["mean_dice"])
    LOGGER.info("mIoU: %.4f", results["metrics"]["mean_miou"])

    attr_accuracy = results.get("attribute_accuracy", {})
    if attr_accuracy:
        LOGGER.info("Attribute accuracy:")
        for slot, info in attr_accuracy.items():
            LOGGER.info(
                "  %s: %.2f%% (%d/%d)",
                slot,
                info["accuracy"] * 100.0,
                info["correct"],
                info["total"],
            )

    LOGGER.info("Outputs saved to: %s", output_dir)
    LOGGER.info("%s", "=" * 80)


if __name__ == "__main__":
    main()
