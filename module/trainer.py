"""Trainer for ReCAP-Seg.

This module provides the main training loop, validation loop, checkpointing,
TensorBoard logging, mixed-precision training, and optional early stopping.

The trainer expects batches with the following keys:
    image: Tensor[B, C, H, W]
    label: Tensor[B, 1, H, W] or Tensor[B, H, W]
    labels: optional structured attribute labels
    filename: optional image names
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


LOGGER = logging.getLogger("ReCAP-Seg")


class AverageMeter:
    """Track the current value, sum, count, and average of a metric."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        val = float(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class EarlyStopping:
    """Early stopping based on a monitored validation score."""

    def __init__(self, patience: int = 100, min_delta: float = 1e-3, mode: str = "max") -> None:
        if mode not in {"max", "min"}:
            raise ValueError(f"Unsupported early-stopping mode: {mode}")

        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        score = float(score)
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        self.early_stop = self.counter >= self.patience
        return self.early_stop


class ReCAPSegTrainer:
    """Training manager for ReCAP-Seg."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str | torch.device = "cuda",
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(device)

        train_cfg = config.get("TRAIN", {})
        self.epochs = int(train_cfg.get("epochs", 500))
        self.accumulation_steps = int(train_cfg.get("accumulation_steps", 1))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.val_threshold = float(train_cfg.get("val_threshold", 0.5))

        self.optimizer = self._create_optimizer(train_cfg)
        self.scheduler = self._create_scheduler(train_cfg)

        self.use_amp = bool(train_cfg.get("use_amp", True)) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        log_cfg = config.get("LOG", {})
        self.log_dir = Path(log_cfg.get("log_dir", "logs/recapseg"))
        self.save_dir = Path(log_cfg.get("save_dir", "checkpoints/recapseg"))
        self.log_interval = int(log_cfg.get("log_interval", 10))
        self.save_interval = int(log_cfg.get("save_interval", 10))
        self.keep_epoch_checkpoints = bool(log_cfg.get("keep_epoch_checkpoints", True))

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir / "tensorboard"))

        early_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=int(early_cfg.get("patience", 100)),
            min_delta=float(early_cfg.get("min_delta", 1e-3)),
            mode=early_cfg.get("mode", "max"),
        )

        self.current_epoch = 0
        self.global_step = 0
        self.best_dice = 0.0
        self.best_epoch = 0

        LOGGER.info("Trainer initialized")
        LOGGER.info("  Epochs   : %d", self.epochs)
        LOGGER.info("  Log dir  : %s", self.log_dir)
        LOGGER.info("  Save dir : %s", self.save_dir)
        LOGGER.info("  AMP      : %s", self.use_amp)

    def _create_optimizer(self, train_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
        """Create optimizer with module-wise parameter groups."""
        optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
        lr = float(train_cfg.get("base_lr", 1e-4))
        weight_decay = float(train_cfg.get("weight_decay", 1e-2))

        encoder_params = []
        prototype_params = []
        decoder_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            lower_name = name.lower()
            if "encoder" in lower_name:
                encoder_params.append(param)
            elif any(key in lower_name for key in ["proto", "prototype", "capb"]):
                prototype_params.append(param)
            elif "decoder" in lower_name or "rrd" in lower_name:
                decoder_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {
                "params": encoder_params,
                "lr": lr * float(train_cfg.get("encoder_lr_mult", 0.1)),
                "weight_decay": weight_decay,
                "name": "encoder",
            },
            {
                "params": prototype_params,
                "lr": lr * float(train_cfg.get("prototype_lr_mult", 2.0)),
                "weight_decay": float(train_cfg.get("prototype_weight_decay", 0.0)),
                "name": "prototype",
            },
            {
                "params": decoder_params,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "decoder",
            },
            {
                "params": other_params,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "other",
            },
        ]
        param_groups = [group for group in param_groups if len(group["params"]) > 0]

        if not param_groups:
            raise ValueError("No trainable parameters found in the model.")

        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999))
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        LOGGER.info("Optimizer: %s", optimizer_name)
        for group in param_groups:
            LOGGER.info(
                "  %s: lr=%.2e, params=%d",
                group.get("name", "unknown"),
                group["lr"],
                len(group["params"]),
            )

        return optimizer

    def _create_scheduler(self, train_cfg: Dict[str, Any]):
        """Create learning-rate scheduler."""
        scheduler_name = str(train_cfg.get("scheduler", "cosine")).lower()
        min_lr = float(train_cfg.get("min_lr", 1e-6))
        self.warmup_epochs = int(train_cfg.get("warmup_epochs", 0))

        if scheduler_name == "none":
            scheduler = None
        elif scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(self.epochs, 1),
                eta_min=min_lr,
            )
        elif scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(train_cfg.get("step_size", 30)),
                gamma=float(train_cfg.get("gamma", 0.5)),
            )
        elif scheduler_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=float(train_cfg.get("factor", 0.5)),
                patience=int(train_cfg.get("scheduler_patience", 10)),
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        LOGGER.info("Scheduler: %s", scheduler_name)
        return scheduler

    def train(self) -> None:
        """Run the full training procedure."""
        LOGGER.info("%s", "=" * 80)
        LOGGER.info("Starting training for %d epochs", self.epochs)
        LOGGER.info("Train batches: %d", len(self.train_loader))
        LOGGER.info("Val batches  : %d", len(self.val_loader))
        LOGGER.info("Train samples: %d", len(self.train_loader.dataset))
        LOGGER.info("Val samples  : %d", len(self.val_loader.dataset))
        LOGGER.info("%s", "=" * 80)

        try:
            for epoch in range(self.current_epoch, self.epochs):
                self.current_epoch = epoch
                if hasattr(self.model, "set_epoch"):
                    self.model.set_epoch(epoch)

                train_metrics = self.train_epoch()
                val_metrics = self.validate()

                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics["dice"])
                    else:
                        self.scheduler.step()

                is_best = val_metrics["dice"] > self.best_dice
                if is_best:
                    self.best_dice = val_metrics["dice"]
                    self.best_epoch = epoch

                self._print_epoch_results(epoch, train_metrics, val_metrics, is_best)
                self._log_epoch(epoch, train_metrics, val_metrics)

                if (epoch + 1) % self.save_interval == 0 or is_best:
                    self.save_checkpoint(epoch, is_best=is_best)

                if self.early_stopping(val_metrics["dice"]):
                    LOGGER.info("Early stopping triggered at epoch %d", epoch)
                    break

            LOGGER.info("%s", "=" * 80)
            LOGGER.info("Training completed")
            LOGGER.info("Best Dice: %.4f at epoch %d", self.best_dice, self.best_epoch)
            LOGGER.info("%s", "=" * 80)
        finally:
            self.writer.close()

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        meters = {
            "loss": AverageMeter(),
            "seg_plain": AverageMeter(),
            "seg_guided": AverageMeter(),
            "align": AverageMeter(),
            "dice": AverageMeter(),
        }

        self.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}", ncols=100)
        pending_update = False

        for step, batch in enumerate(pbar):
            batch = self._to_device(batch)

            with autocast(enabled=self.use_amp):
                output = self.model(batch)
                if "loss" not in output:
                    raise KeyError("Model output must contain a 'loss' key during training.")
                loss = output["loss"] / self.accumulation_steps

            self._check_finite_loss(loss, output, batch, step)

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            pending_update = True
            should_step = (step + 1) % self.accumulation_steps == 0
            if should_step:
                self._optimizer_step()
                pending_update = False

            batch_size = batch["image"].size(0)
            loss_value = float(loss.item()) * self.accumulation_steps
            meters["loss"].update(loss_value, batch_size)
            self._update_loss_component_meters(meters, output)

            with torch.no_grad():
                pred = torch.sigmoid(self._ensure_mask_shape(output["logits"])) > 0.5
                target = self._ensure_mask_shape(batch["label"]) > 0.5
                dice = self._compute_dice(pred, target)
                meters["dice"].update(float(dice.item()), batch_size)

            pbar.set_postfix({"loss": f"{meters['loss'].avg:.4f}", "dice": f"{meters['dice'].avg:.4f}"})

            if self.global_step % self.log_interval == 0:
                self.writer.add_scalar("train/loss", loss_value, self.global_step)
                self.writer.add_scalar("train/dice", meters["dice"].avg, self.global_step)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)

        if pending_update:
            self._optimizer_step()

        return {key: meter.avg for key, meter in meters.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model using two-stage inference when available."""
        self.model.eval()

        meters = {
            "dice": AverageMeter(),
            "miou": AverageMeter(),
        }
        attr_correct: Dict[str, int] = {}
        attr_total: Dict[str, int] = {}

        for batch in tqdm(self.val_loader, desc="Validating", ncols=100):
            batch = self._to_device(batch)
            output = self._run_inference(batch["image"])

            pred = torch.sigmoid(self._ensure_mask_shape(output["logits"])) > self.val_threshold
            target = self._ensure_mask_shape(batch["label"]) > 0.5

            dice = self._compute_dice(pred, target)
            miou = self._compute_miou(pred, target)

            batch_size = batch["image"].size(0)
            meters["dice"].update(float(dice.item()), batch_size)
            meters["miou"].update(float(miou.item()), batch_size)

            if "pred_attributes" in output and "labels" in batch:
                self._update_attr_accuracy(output["pred_attributes"], batch["labels"], attr_correct, attr_total)

        metrics: Dict[str, float] = {key: meter.avg for key, meter in meters.items()}
        for slot, correct in attr_correct.items():
            total = attr_total.get(slot, 0)
            if total > 0:
                metrics[f"attr_{slot}"] = correct / total

        return metrics

    def _run_inference(self, images: torch.Tensor) -> Dict[str, Any]:
        """Run model inference with support for ReCAP-Seg two-stage inference."""
        if hasattr(self.model, "inference_two_stage"):
            return self.model.inference_two_stage(
                images,
                threshold=self.val_threshold,
                return_intermediate=True,
            )

        output = self.model(images)
        if isinstance(output, dict):
            return output
        return {"logits": output}

    def _optimizer_step(self) -> None:
        """Apply optimizer step with optional AMP and gradient clipping."""
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move tensor fields in a batch to the target device."""
        for key in ["image", "label"]:
            if key in batch and isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device, non_blocking=True)
        return batch

    @staticmethod
    def _ensure_mask_shape(mask: torch.Tensor) -> torch.Tensor:
        """Ensure a tensor has shape [B, 1, H, W]."""
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4:
            raise ValueError(f"Expected mask shape [B, 1, H, W] or [B, H, W], got {tuple(mask.shape)}")
        return mask

    @staticmethod
    def _compute_dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Compute batch-averaged Dice score."""
        pred = pred.float().flatten(1)
        target = target.float().flatten(1)
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + eps) / (union + eps)
        return dice.mean()

    @staticmethod
    def _compute_miou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Compute batch-averaged mIoU score."""
        pred = pred.float().flatten(1)
        target = target.float().flatten(1)
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1) - intersection
        iou = (intersection + eps) / (union + eps)
        return iou.mean()

    @staticmethod
    def _update_attr_accuracy(
        pred_attrs: Dict[str, Dict[str, Any]],
        gt_labels: List[Optional[Dict[str, List[str]]]],
        correct: Dict[str, int],
        total: Dict[str, int],
    ) -> None:
        """Update slot-wise attribute accuracy when attribute labels are available."""
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

    @staticmethod
    def _tensor_to_float(value: Any) -> Optional[float]:
        """Convert a tensor or scalar to float for logging."""
        if isinstance(value, torch.Tensor):
            return float(value.detach().item())
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _update_loss_component_meters(self, meters: Dict[str, AverageMeter], output: Dict[str, Any]) -> None:
        """Update optional loss component meters."""
        components = output.get("loss_components", {})
        component_map = {
            "seg_plain": "seg_plain",
            "seg_guided": "seg_guided",
            "align_total": "align",
            "align": "align",
        }
        for component_key, meter_key in component_map.items():
            if component_key in components:
                value = self._tensor_to_float(components[component_key])
                if value is not None:
                    meters[meter_key].update(value)

    def _check_finite_loss(self, loss: torch.Tensor, output: Dict[str, Any], batch: Dict[str, Any], step: int) -> None:
        """Stop training when NaN/Inf loss is detected."""
        if torch.isfinite(loss):
            return

        LOGGER.error("NaN or Inf loss detected at epoch %d, step %d", self.current_epoch, step)
        full_loss = output.get("loss")
        if isinstance(full_loss, torch.Tensor):
            LOGGER.error("Total loss: %s", full_loss.detach().cpu().item())

        components = output.get("loss_components", {})
        for key, value in components.items():
            LOGGER.error("Loss component %s: %s", key, self._tensor_to_float(value))

        if "logits" in output and isinstance(output["logits"], torch.Tensor):
            logits = output["logits"].detach()
            LOGGER.error(
                "Logits stats: min=%.4e, max=%.4e, mean=%.4e",
                logits.min().item(),
                logits.max().item(),
                logits.mean().item(),
            )

        label = batch.get("label")
        if isinstance(label, torch.Tensor):
            LOGGER.error(
                "Label stats: min=%.4e, max=%.4e, mean=%.4e",
                label.min().item(),
                label.max().item(),
                label.mean().item(),
            )

        raise RuntimeError("NaN or Inf loss encountered; training aborted.")

    def _print_epoch_results(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        is_best: bool,
    ) -> None:
        """Log epoch-level results."""
        LOGGER.info("Epoch %d/%d", epoch, self.epochs - 1)
        LOGGER.info("  Train - Loss: %.4f, Dice: %.4f", train_metrics["loss"], train_metrics["dice"])
        LOGGER.info("  Val   - Dice: %.4f, mIoU: %.4f", val_metrics["dice"], val_metrics["miou"])

        attr_items = [
            f"{key.replace('attr_', '')}: {value:.2%}"
            for key, value in val_metrics.items()
            if key.startswith("attr_")
        ]
        if attr_items:
            LOGGER.info("  Attrs - %s", ", ".join(attr_items))

        LOGGER.info("  LR: %.2e", self.optimizer.param_groups[0]["lr"])
        if is_best:
            LOGGER.info("  New best Dice")

    def _log_epoch(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """Write epoch-level metrics to TensorBoard."""
        for key, value in train_metrics.items():
            self.writer.add_scalar(f"train_epoch/{key}", value, epoch)

        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"val/{key}", value, epoch)

        self.writer.add_scalar("train_epoch/lr", self.optimizer.param_groups[0]["lr"], epoch)

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save latest, optional epoch, and best checkpoints."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_dice": self.best_dice,
            "best_epoch": self.best_epoch,
            "config": self.config,
        }

        latest_path = self.save_dir / "latest.pth"
        torch.save(checkpoint, latest_path)

        if self.keep_epoch_checkpoints:
            epoch_path = self.save_dir / f"epoch_{epoch:03d}.pth"
            torch.save(checkpoint, epoch_path)

        if is_best:
            best_path = self.save_dir / "best.pth"
            torch.save(checkpoint, best_path)
            LOGGER.info("Saved best checkpoint: %s", best_path)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Resume training from a saved checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        LOGGER.info("Loading checkpoint: %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            cleaned_key = key.replace("module.", "", 1) if key.startswith("module.") else key
            cleaned_state_dict[cleaned_key] = value

        self.model.load_state_dict(cleaned_state_dict, strict=True)

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if checkpoint.get("scheduler_state_dict") is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = int(checkpoint.get("epoch", -1)) + 1
        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_dice = float(checkpoint.get("best_dice", 0.0))
        self.best_epoch = int(checkpoint.get("best_epoch", 0))

        LOGGER.info(
            "Resumed from epoch %d; best Dice %.4f at epoch %d",
            self.current_epoch - 1,
            self.best_dice,
            self.best_epoch,
        )

