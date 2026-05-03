"""Loss functions for ReCAP-Seg.

This module contains segmentation losses and prototype-alignment losses used by
ReCAP-Seg. The main public class is ``ReCAPSegLoss``. A backward-compatible alias
``ProLearnV7Loss`` is provided for older internal scripts.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss for class-imbalanced attribute prediction.

    Args:
        alpha: Optional class weights with shape ``[C]``.
        gamma: Focusing parameter.
        reduction: ``mean``, ``sum``, or ``none``.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")

        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Logits with shape ``[N, C]``.
            targets: Class indices with shape ``[N]``.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = (1.0 - pt).pow(self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MultiLabelFocalLoss(nn.Module):
    """Soft multi-label focal loss for slots with multiple valid categories.

    Args:
        gamma: Focusing parameter.
        reduction: ``mean``, ``sum``, or ``none``.
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute multi-label focal loss.

        Args:
            inputs: Logits with shape ``[N, C]``.
            targets: Soft labels with shape ``[N, C]``. Each row may be one-hot,
                multi-hot, or probability-normalized.
        """
        if inputs.shape != targets.shape:
            raise ValueError(f"inputs and targets must have the same shape, got {inputs.shape} and {targets.shape}")

        log_probs = F.log_softmax(inputs, dim=-1)
        probs = log_probs.exp()
        focal_weight = (1.0 - probs).pow(self.gamma)
        loss = -(targets * focal_weight * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BoundaryLoss(nn.Module):
    """Boundary-aware binary loss based on Sobel edge extraction.

    The input prediction should be a probability map after sigmoid. The target
    should be a binary mask.
    """

    def __init__(self, edge_threshold: float = 0.1, eps: float = 1e-7) -> None:
        super().__init__()
        self.edge_threshold = float(edge_threshold)
        self.eps = float(eps)

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    @staticmethod
    def _ensure_mask_shape(mask: torch.Tensor) -> torch.Tensor:
        """Ensure tensor shape is ``[B, 1, H, W]``."""
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4:
            raise ValueError(f"Expected shape [B, 1, H, W] or [B, H, W], got {tuple(mask.shape)}")
        return mask.float()

    def get_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """Extract binary boundary map from a mask."""
        mask = self._ensure_mask_shape(mask)
        mask = F.pad(mask, (1, 1, 1, 1), mode="replicate")
        edge_x = F.conv2d(mask, self.sobel_x, padding=0)
        edge_y = F.conv2d(mask, self.sobel_y, padding=0)
        boundary = torch.sqrt(edge_x.pow(2) + edge_y.pow(2) + 1e-8)
        return (boundary > self.edge_threshold).float()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute boundary loss.

        Args:
            pred: Sigmoid probabilities with shape ``[B, 1, H, W]``.
            target: Binary target mask with shape ``[B, 1, H, W]``.
        """
        pred = self._ensure_mask_shape(pred)
        target = self._ensure_mask_shape(target)
        boundary = self.get_boundary(target)

        if boundary.sum() < 1e-6:
            return pred.new_zeros(())

        pred_boundary = (pred * boundary).clamp(self.eps, 1.0 - self.eps)
        target_boundary = target * boundary
        bce = -target_boundary * torch.log(pred_boundary) - (1.0 - target_boundary) * torch.log(1.0 - pred_boundary)
        return bce.sum() / (boundary.sum() + self.eps)


class BinaryDiceBCELoss(nn.Module):
    """Binary Dice + BCE loss for segmentation logits.

    This implementation avoids an additional MONAI dependency while matching the
    common Dice+CE formulation used in medical image segmentation.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        weight_sum = float(dice_weight + bce_weight)
        if weight_sum <= 0:
            raise ValueError("dice_weight + bce_weight must be positive.")
        self.dice_weight = float(dice_weight) / weight_sum
        self.bce_weight = float(bce_weight) / weight_sum
        self.smooth = float(smooth)

    @staticmethod
    def _ensure_mask_shape(mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4:
            raise ValueError(f"Expected shape [B, 1, H, W] or [B, H, W], got {tuple(mask.shape)}")
        return mask.float()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice+BCE loss from logits."""
        logits = self._ensure_mask_shape(logits)
        target = self._ensure_mask_shape(target)

        probs = torch.sigmoid(logits)
        probs_flat = probs.flatten(1)
        target_flat = target.flatten(1)

        intersection = (probs_flat * target_flat).sum(dim=1)
        denominator = probs_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice.mean()

        bce_loss = F.binary_cross_entropy_with_logits(logits, target)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class SegmentationLoss(nn.Module):
    """Segmentation loss combining Dice+BCE and boundary loss.

    Args:
        dice_weight: Relative Dice weight inside Dice+BCE.
        ce_weight: Relative BCE weight inside Dice+BCE. Kept as ``ce_weight`` for
            compatibility with earlier configs.
        boundary_weight: Weight of the boundary loss.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        boundary_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.boundary_weight = float(boundary_weight)
        self.dice_bce = BinaryDiceBCELoss(dice_weight=dice_weight, bce_weight=ce_weight)
        self.boundary_loss = BoundaryLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """Compute segmentation loss from logits and target masks."""
        dice_bce = self.dice_bce(pred, target)
        boundary = self.boundary_loss(torch.sigmoid(pred), target)
        total = dice_bce + self.boundary_weight * boundary

        if return_components:
            return {
                "total": total,
                "dice_ce": dice_bce,
                "boundary": boundary,
            }
        return total


class AttributeAlignmentLoss(nn.Module):
    """Prototype alignment loss for lesion embeddings and attribute prototypes."""

    def __init__(
        self,
        temperature: float = 0.07,
        use_focal: bool = True,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.use_focal = bool(use_focal)
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.multilabel_focal = MultiLabelFocalLoss(gamma=focal_gamma)

    def forward(
        self,
        embeddings: torch.Tensor,
        prototypes: torch.Tensor,
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        is_multilabel: bool = False,
    ) -> torch.Tensor:
        """Compute slot-wise alignment loss."""
        if embeddings.numel() == 0 or embeddings.shape[0] == 0:
            return embeddings.new_zeros(())

        embeddings = F.normalize(embeddings, dim=-1)
        prototypes = F.normalize(prototypes, dim=-1)
        logits = embeddings @ prototypes.t() / self.temperature

        if is_multilabel:
            return self.multilabel_focal(logits, labels.float())

        if class_weights is not None:
            class_weights = class_weights.to(logits.device)
            ce_loss = F.cross_entropy(logits, labels.long(), weight=class_weights)
            if not self.use_focal:
                return ce_loss

        if self.use_focal:
            return FocalLoss(alpha=class_weights, gamma=self.focal_loss.gamma)(logits, labels.long())

        return F.cross_entropy(logits, labels.long(), weight=class_weights)

    def forward_with_infonce(
        self,
        embeddings: torch.Tensor,
        prototypes: torch.Tensor,
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        focal_weight: float = 0.7,
    ) -> torch.Tensor:
        """Combine prototype classification loss and sample-level InfoNCE."""
        if embeddings.numel() == 0 or embeddings.shape[0] == 0:
            return embeddings.new_zeros(())

        embeddings = F.normalize(embeddings, dim=-1)
        prototypes = F.normalize(prototypes, dim=-1)
        labels = labels.long()

        logits_cls = embeddings @ prototypes.t() / self.temperature
        focal = FocalLoss(alpha=class_weights, gamma=self.focal_loss.gamma)(logits_cls, labels)

        # InfoNCE needs at least two samples to define negative pairs.
        if embeddings.shape[0] < 2:
            return focal

        positive_prototypes = prototypes[labels]
        sim_matrix = embeddings @ positive_prototypes.t() / self.temperature
        targets = torch.arange(embeddings.shape[0], device=embeddings.device)
        infonce = F.cross_entropy(sim_matrix, targets)

        focal_weight = float(focal_weight)
        return focal_weight * focal + (1.0 - focal_weight) * infonce


class ReCAPSegLoss(nn.Module):
    """Combined loss for ReCAP-Seg.

    The class exposes ``seg_loss`` and ``compute_alignment_losses`` because the
    main model computes segmentation, attribute-classification, and alignment
    terms in a staged manner.
    """

    INVALID_LABELS = {"unknown", "not detected", "invalid", "none", "null", ""}

    def __init__(
        self,
        seg_weight: float = 1.0,
        align_weight: float = 1.0,
        guided_weight: float = 0.5,
        boundary_weight: float = 0.2,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.seg_weight = float(seg_weight)
        self.align_weight = float(align_weight)
        self.guided_weight = float(guided_weight)
        self.boundary_weight = float(boundary_weight)

        self.seg_loss = SegmentationLoss(boundary_weight=boundary_weight)
        self.align_loss = AttributeAlignmentLoss(temperature=temperature)

    def compute_alignment_losses(
        self,
        lesion_emb: torch.Tensor,
        proto_lib: Any,
        labels_list: List[Optional[Dict[str, List[str]]]],
        slot_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute prototype alignment loss for all attribute slots."""
        device = lesion_emb.device
        total_loss = lesion_emb.new_zeros(())
        loss_dict: Dict[str, float] = {}
        valid_slot_count = 0

        for slot in proto_lib.get_slot_names():
            valid_indices: List[int] = []
            valid_labels: List[int | torch.Tensor] = []
            is_multilabel = False
            num_classes = proto_lib.get_num_values(slot)

            for sample_idx, label_dict in enumerate(labels_list):
                if label_dict is None or slot not in label_dict:
                    continue

                values = label_dict[slot]
                if not values:
                    continue

                if str(values[0]).lower() in self.INVALID_LABELS:
                    continue

                indices = proto_lib.labels_to_indices(slot, values)
                if not indices:
                    continue

                valid_indices.append(sample_idx)
                if len(indices) > 1:
                    is_multilabel = True
                    soft_label = torch.zeros(num_classes, device=device)
                    for idx in indices:
                        soft_label[idx] = 1.0 / len(indices)
                    valid_labels.append(soft_label)
                else:
                    valid_labels.append(indices[0])

            if not valid_indices:
                loss_dict[f"{slot}_align"] = 0.0
                continue

            valid_emb = lesion_emb[valid_indices]
            prototypes = proto_lib.get_prototypes(slot)

            if is_multilabel:
                label_vectors = []
                for label in valid_labels:
                    if isinstance(label, torch.Tensor):
                        label_vectors.append(label.to(device=device, dtype=torch.float32))
                    else:
                        one_hot = torch.zeros(num_classes, device=device)
                        one_hot[int(label)] = 1.0
                        label_vectors.append(one_hot)
                labels_tensor = torch.stack(label_vectors, dim=0)
                slot_loss = self.align_loss(valid_emb, prototypes, labels_tensor, is_multilabel=True)
            else:
                labels_tensor = torch.tensor(valid_labels, device=device, dtype=torch.long)
                slot_loss = self.align_loss.forward_with_infonce(valid_emb, prototypes, labels_tensor)

            weight = slot_weights.get(slot, 1.0) if slot_weights is not None else proto_lib.get_slot_weight(slot)
            total_loss = total_loss + float(weight) * slot_loss
            loss_dict[f"{slot}_align"] = float(slot_loss.detach().item())
            valid_slot_count += 1

        if valid_slot_count > 0:
            total_loss = total_loss / valid_slot_count

        return total_loss, loss_dict

    def forward(
        self,
        plain_logits: torch.Tensor,
        guided_logits: Optional[torch.Tensor],
        target: torch.Tensor,
        lesion_emb: torch.Tensor,
        proto_lib: Any,
        labels_list: List[Optional[Dict[str, List[str]]]],
        epoch: int = 0,
        slot_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Compute combined segmentation and alignment loss.

        This method is kept for standalone use. The current ``ReCAPSeg`` model
        may also call ``seg_loss`` and ``compute_alignment_losses`` separately.
        """
        loss_dict: Dict[str, float] = {}

        seg_plain = self.seg_loss(plain_logits, target, return_components=True)
        total_seg_loss = seg_plain["total"]
        loss_dict["seg_plain"] = float(seg_plain["total"].detach().item())
        loss_dict["seg_plain_dice_ce"] = float(seg_plain["dice_ce"].detach().item())
        loss_dict["seg_plain_boundary"] = float(seg_plain["boundary"].detach().item())

        if guided_logits is not None:
            seg_guided = self.seg_loss(guided_logits, target, return_components=True)
            guided_weight = self._get_guided_weight(epoch)
            total_seg_loss = total_seg_loss + guided_weight * seg_guided["total"]
            loss_dict["seg_guided"] = float(seg_guided["total"].detach().item())
            loss_dict["guided_weight"] = guided_weight

        align_loss, align_dict = self.compute_alignment_losses(
            lesion_emb,
            proto_lib,
            labels_list,
            slot_weights=slot_weights,
        )
        loss_dict.update(align_dict)
        loss_dict["align_total"] = float(align_loss.detach().item())

        align_weight = self._get_align_weight(epoch)
        loss_dict["align_weight_dynamic"] = align_weight

        total_loss = self.seg_weight * total_seg_loss + align_weight * self.align_weight * align_loss
        loss_dict["total"] = float(total_loss.detach().item())

        return {"loss": total_loss, "components": loss_dict}

    @staticmethod
    def _get_guided_weight(epoch: int) -> float:
        """Dynamic weight for guided-branch segmentation loss."""
        if epoch <= 30:
            return 0.3
        if epoch <= 60:
            return 0.5
        if epoch <= 120:
            return 0.8
        return 1.0

    @staticmethod
    def _get_align_weight(epoch: int) -> float:
        """Dynamic weight for prototype alignment loss."""
        if epoch <= 30:
            return 1.0
        if epoch <= 60:
            return 0.8
        if epoch <= 120:
            return 0.6
        return 0.4


