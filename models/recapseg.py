"""ReCAP-Seg model.

ReCAP-Seg is a prompt-free medical image segmentation framework that learns from
image-mask-attribute triplets during training and performs image-only inference
at test time. The model uses a plain visual branch to obtain a coarse mask,
retrieves slot-wise clinical attribute priors from a prototype bank, and refines
segmentation through a retrieval-conditioned decoder.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attribute_prototype import AttributePrototypeLibrary
from .convnext_encoder import ConvNeXtV2Encoder
from .cross_attention_decoder import DualBranchDecoder
from .roi_pooling import MultiScaleMaskROIPooling

from .losses import ReCAPSegLoss



LOGGER = logging.getLogger("ReCAP-Seg")


class ReCAPSeg(nn.Module):
    """Prompt-free segmentation via retrievable clinical attribute priors.

    Training:
        1. Encode image into multi-scale features.
        2. Use the ground-truth mask to obtain a lesion-centric embedding.
        3. Align lesion embeddings with slot-wise attribute prototypes.
        4. Inject training-time attribute embeddings into the guided decoder.

    Inference:
        1. Predict a coarse mask with the plain visual branch.
        2. Use the coarse mask as a pseudo ROI for lesion embedding extraction.
        3. Retrieve slot-wise attribute priors from the prototype bank.
        4. Refine segmentation with the retrieval-conditioned decoder.

    Args:
        args: Optional object containing runtime attributes such as ``device``.
        encoder_name: Encoder backbone name.
        embed_dim: Dimension of lesion and attribute embeddings.
        frozen_stages: Number of frozen encoder stages.
        num_heads: Number of attention heads used in the guided decoder.
        use_text_init: Whether to initialize attribute prototypes from text.
        attribute_config: Optional task-specific attribute schema.
        out_size: Output segmentation size as ``(H, W)``.
        pretrained_encoder: Whether to load pretrained encoder weights.
        guided_start_epoch: Epoch from which the guided branch is used.
        align_start_epoch: Epoch from which prototype alignment loss is used.
    """

    def __init__(
        self,
        args: Optional[Any] = None,
        encoder_name: str = "convnextv2_base",
        embed_dim: int = 512,
        frozen_stages: int = 0,
        num_heads: int = 8,
        use_text_init: bool = True,
        attribute_config: Optional[Dict[str, Any]] = None,
        out_size: Tuple[int, int] = (256, 256),
        pretrained_encoder: bool = True,
        guided_start_epoch: int = 10,
        align_start_epoch: int = 20,
    ) -> None:
        super().__init__()

        self.args = args
        self.embed_dim = int(embed_dim)
        self.device_name = str(getattr(args, "device", "cuda")) if args is not None else "cuda"
        self.current_epoch = 0
        self.global_step = 0
        self.guided_start_epoch = int(guided_start_epoch)
        self.align_start_epoch = int(align_start_epoch)

        self.encoder = ConvNeXtV2Encoder(
            model_name=encoder_name,
            pretrained=pretrained_encoder,
            frozen_stages=frozen_stages,
            output_dim=self.embed_dim,
        )
        self.feature_dims = self.encoder.get_feature_dims()

        self.prototype_lib = AttributePrototypeLibrary(
            embed_dim=self.embed_dim,
            attribute_config=attribute_config,
            use_text_init=use_text_init,
        )
        # Backward-compatible alias used by older internal code.
        self.proto_lib = self.prototype_lib
        self.num_slots = self.prototype_lib.get_num_slots()

        self.roi_pool = MultiScaleMaskROIPooling(
            feature_dims=self.feature_dims,
            output_dim=self.embed_dim,
            pool_type="weighted_avg",
            fusion_type="concat_proj",
        )

        self.decoder = DualBranchDecoder(
            feature_dims=self.feature_dims,
            attr_dim=self.embed_dim,
            num_heads=num_heads,
            out_channels=1,
            out_size=out_size,
            fusion_type="learnable",
        )

        self.loss_fn = ReCAPSegLoss(
            seg_weight=1.0,
            align_weight=1.0,
            guided_weight=0.5,
            boundary_weight=0.2,
        )

        self.attr_classifiers = self._create_attribute_classifiers()

        LOGGER.info("ReCAP-Seg initialized")
        LOGGER.info("  Encoder      : %s", encoder_name)
        LOGGER.info("  Feature dims : %s", self.feature_dims)
        LOGGER.info("  Embed dim    : %d", self.embed_dim)
        LOGGER.info("  Num slots    : %d", self.num_slots)
        LOGGER.info("  Output size  : %s", out_size)

    def _create_attribute_classifiers(self) -> nn.ModuleDict:
        """Create slot-wise attribute classification heads."""
        classifiers = nn.ModuleDict()
        hidden_dim = max(self.embed_dim // 2, 1)

        for slot in self.prototype_lib.get_slot_names():
            num_classes = self.prototype_lib.get_num_values(slot)
            classifiers[slot] = nn.Sequential(
                nn.Linear(self.embed_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(hidden_dim, num_classes),
            )

        return classifiers

    def encode_image(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode an input image into multi-scale visual features."""
        return self.encoder(image)

    def extract_lesion_embedding(
        self,
        features: List[torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract a normalized lesion-centric embedding using mask-guided pooling."""
        mask = self._ensure_mask_shape(mask)
        lesion_embedding = self.roi_pool(features, mask)
        return F.normalize(lesion_embedding, dim=-1)

    def build_attribute_embedding(
        self,
        labels_list: List[Optional[Dict[str, List[str]]]],
    ) -> torch.Tensor:
        """Build slot-wise attribute embeddings from structured labels."""
        try:
            return self.prototype_lib.build_batch_attribute_embeddings(
                labels_list,
                device=self._module_device,
            )
        except TypeError:
            return self.prototype_lib.build_batch_attribute_embeddings(labels_list)

    def retrieve_attribute_embedding(
        self,
        lesion_embedding: torch.Tensor,
        soft: bool = True,
    ) -> torch.Tensor:
        """Retrieve slot-wise attribute embeddings from lesion embeddings."""
        return self.prototype_lib.retrieve_all_slots(lesion_embedding, soft=soft)

    def classify_attributes(self, lesion_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return slot-wise attribute logits from lesion embeddings."""
        return {
            slot: classifier(lesion_embedding)
            for slot, classifier in self.attr_classifiers.items()
        }

    def predict_attributes(self, lesion_embedding: torch.Tensor) -> Dict[str, Dict[str, Any]]:
        """Predict slot-wise attributes for analysis or optional output."""
        predictions: Dict[str, Dict[str, Any]] = {}
        attr_logits = self.classify_attributes(lesion_embedding)

        for slot, logits in attr_logits.items():
            probs = F.softmax(logits, dim=-1)
            pred_idx = probs.argmax(dim=-1)
            pred_values = [
                self.prototype_lib.idx_to_label(slot, int(idx.item()))
                for idx in pred_idx
            ]

            predictions[slot] = {
                "logits": logits,
                "probs": probs,
                "pred_idx": pred_idx,
                "pred_values": pred_values,
            }

        return predictions

    def segment(
        self,
        features: List[torch.Tensor],
        attr_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the dual-branch segmentation decoder."""
        return self.decoder(features, attr_embedding)

    @torch.no_grad()
    def inference_two_stage(
        self,
        image: torch.Tensor,
        threshold: float = 0.5,
        return_intermediate: bool = False,
    ) -> Dict[str, Any]:
        """Run prompt-free two-stage inference.

        Args:
            image: Input tensor with shape ``[B, 3, H, W]``.
            threshold: Threshold used to binarize the plain-branch coarse mask.
            return_intermediate: Whether to return pseudo masks, embeddings, and
                attribute predictions.

        Returns:
            Dictionary containing final logits and optional intermediate outputs.
        """
        enc_out = self.encode_image(image)
        features = enc_out["multi_scale_features"]

        stage1_logits = self.decoder.forward_plain_only(features)
        pseudo_mask = (torch.sigmoid(stage1_logits) > threshold).float()

        lesion_embedding = self.extract_lesion_embedding(features, pseudo_mask)
        attr_embedding = self.retrieve_attribute_embedding(lesion_embedding, soft=True)

        stage2_logits = self.decoder.forward_guided_only(features, attr_embedding)
        seg_out = self.decoder(features, attr_embedding)

        output: Dict[str, Any] = {
            "logits": seg_out["logits"],
            "stage1_logits": stage1_logits,
            "stage2_logits": stage2_logits,
        }

        if "plain_logits" in seg_out:
            output["plain_logits"] = seg_out["plain_logits"]
        if "guided_logits" in seg_out:
            output["guided_logits"] = seg_out["guided_logits"]

        if return_intermediate:
            output.update(
                {
                    "pseudo_mask": pseudo_mask,
                    "lesion_emb": lesion_embedding,
                    "attr_emb": attr_embedding,
                    "pred_attributes": self.predict_attributes(lesion_embedding),
                }
            )

        return output

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Run training forward propagation.

        Args:
            batch: Dictionary containing ``image``, optional ``label``, and
                optional structured attribute ``labels``.

        Returns:
            Dictionary containing logits, optional loss, and optional
            intermediate representations.
        """
        image = batch["image"]
        mask = batch.get("label")
        labels_list = batch.get("labels")
        self.global_step += 1

        if mask is not None:
            mask = self._ensure_mask_shape(mask.float())

        enc_out = self.encode_image(image)
        features = enc_out["multi_scale_features"]
        global_feature = F.normalize(enc_out["global_feature"], dim=-1)

        plain_logits = self.decoder.forward_plain_only(features)
        lesion_embedding = self._extract_training_embedding(features, mask, global_feature)

        guided_logits = None
        attr_embedding = None
        if self._use_guided_branch(labels_list):
            attr_embedding = self.build_attribute_embedding(labels_list)
            if self._is_finite_tensor(attr_embedding):
                attr_embedding = torch.clamp(attr_embedding, min=-10.0, max=10.0)
                guided_logits = self.decoder.forward_guided_only(features, attr_embedding)
                if not self._is_finite_tensor(guided_logits):
                    LOGGER.warning("Non-finite guided logits detected; falling back to plain logits.")
                    guided_logits = None
                    attr_embedding = None
            else:
                LOGGER.warning("Non-finite attribute embeddings detected; skipping guided branch.")
                attr_embedding = None

        if guided_logits is not None and attr_embedding is not None:
            seg_out = self.decoder(features, attr_embedding)
            logits = seg_out["logits"]
            if not self._is_finite_tensor(logits):
                LOGGER.warning("Non-finite fused logits detected; falling back to plain logits.")
                logits = plain_logits
        else:
            logits = plain_logits

        output: Dict[str, Any] = {
            "logits": logits,
            "plain_logits": plain_logits,
            "guided_logits": guided_logits,
            "lesion_emb": lesion_embedding,
            "global_feat": global_feature,
        }

        if mask is not None:
            loss_output = self.compute_loss(
                plain_logits=plain_logits,
                guided_logits=guided_logits,
                target=mask,
                lesion_embedding=lesion_embedding,
                labels_list=labels_list,
            )
            output["loss"] = loss_output["loss"]
            output["loss_components"] = loss_output["components"]
        else:
            output["loss"] = None
            output["loss_components"] = {}

        with torch.no_grad():
            output["pred_attributes"] = self.predict_attributes(lesion_embedding)

        return output

    def compute_loss(
        self,
        plain_logits: torch.Tensor,
        guided_logits: Optional[torch.Tensor],
        target: torch.Tensor,
        lesion_embedding: torch.Tensor,
        labels_list: Optional[List[Optional[Dict[str, List[str]]]]],
    ) -> Dict[str, Any]:
        """Compute segmentation, attribute classification, and alignment losses."""
        loss_components: Dict[str, float] = {}

        seg_plain = self.loss_fn.seg_loss(plain_logits, target)
        total_loss = seg_plain
        loss_components["seg_plain"] = float(seg_plain.detach().item())

        if guided_logits is not None and self._is_finite_tensor(guided_logits):
            seg_guided = self.loss_fn.seg_loss(guided_logits, target)
            if self._is_finite_tensor(seg_guided):
                guided_weight = self._get_guided_weight()
                total_loss = total_loss + guided_weight * seg_guided
                loss_components["seg_guided"] = float(seg_guided.detach().item())
                loss_components["guided_weight"] = guided_weight
            else:
                loss_components["seg_guided"] = 0.0
                loss_components["guided_weight"] = 0.0

        if labels_list is not None:
            classification_loss, classification_components = self._compute_classification_loss(
                lesion_embedding,
                labels_list,
            )
            classification_weight = self._get_classification_weight()
            total_loss = total_loss + classification_weight * classification_loss
            loss_components.update(classification_components)
            loss_components["cls_weight"] = classification_weight
            loss_components["cls_total"] = float(classification_loss.detach().item())

        if labels_list is not None and self.current_epoch >= self.align_start_epoch:
            align_loss, align_components = self.loss_fn.compute_alignment_losses(
                lesion_embedding,
                self.prototype_lib,
                labels_list,
            )
            align_weight = self._get_align_weight()
            total_loss = total_loss + align_weight * align_loss

            for key, value in align_components.items():
                loss_components[f"align_{key}"] = self._safe_float(value)
            loss_components["align_weight"] = align_weight
            loss_components["align_total"] = float(align_loss.detach().item())

        loss_components["total"] = float(total_loss.detach().item())
        return {"loss": total_loss, "components": loss_components}

    def _compute_classification_loss(
        self,
        lesion_embedding: torch.Tensor,
        labels_list: List[Optional[Dict[str, List[str]]]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute slot-wise attribute classification loss."""
        device = lesion_embedding.device
        attr_logits = self.classify_attributes(lesion_embedding)

        total_loss = torch.zeros((), device=device)
        num_valid_slots = 0
        loss_components: Dict[str, float] = {}

        for slot, logits in attr_logits.items():
            valid_indices: List[int] = []
            targets: List[int] = []

            for sample_idx, label_dict in enumerate(labels_list):
                if label_dict is None or slot not in label_dict:
                    continue

                values = label_dict[slot]
                if not values or values[0] in {"unknown", "not detected", "invalid"}:
                    continue

                indices = self.prototype_lib.labels_to_indices(slot, values)
                if not indices:
                    continue

                valid_indices.append(sample_idx)
                targets.append(indices[0])

            if not valid_indices:
                loss_components[f"{slot}_cls"] = 0.0
                continue

            valid_logits = logits[valid_indices]
            target_tensor = torch.tensor(targets, device=device, dtype=torch.long)
            slot_loss = F.cross_entropy(valid_logits, target_tensor)
            slot_weight = float(self.prototype_lib.get_slot_weight(slot))

            total_loss = total_loss + slot_weight * slot_loss
            num_valid_slots += 1
            loss_components[f"{slot}_cls"] = float(slot_loss.detach().item())

        if num_valid_slots > 0:
            total_loss = total_loss / num_valid_slots

        return total_loss, loss_components

    def _extract_training_embedding(
        self,
        features: List[torch.Tensor],
        mask: Optional[torch.Tensor],
        global_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Extract lesion embedding during training using GT mask when available."""
        if mask is not None and torch.count_nonzero(mask) > 0:
            return self.extract_lesion_embedding(features, mask)
        return global_feature

    def _use_guided_branch(self, labels_list: Optional[List[Any]]) -> bool:
        """Return whether the guided branch should be active in training."""
        if labels_list is None:
            return False
        if self.current_epoch < self.guided_start_epoch:
            return False
        return any(label is not None for label in labels_list)

    def _get_classification_weight(self) -> float:
        """Schedule for the attribute classification loss."""
        epoch = self.current_epoch
        if epoch < 5:
            return 1.0
        if epoch < 20:
            return 1.5
        if epoch < 50:
            return 2.0
        return 1.5

    def _get_guided_weight(self) -> float:
        """Schedule for the guided-branch segmentation loss."""
        epoch = self.current_epoch
        if epoch < self.guided_start_epoch:
            return 0.0
        if epoch < 30:
            return 0.2
        if epoch < 60:
            return 0.4
        return 0.6

    def _get_align_weight(self) -> float:
        """Schedule for the prototype alignment loss."""
        epoch = self.current_epoch
        if epoch < self.align_start_epoch:
            return 0.0
        if epoch < 40:
            return 0.2
        if epoch < 60:
            return 0.3
        return 0.5

    def set_epoch(self, epoch: int) -> None:
        """Update the current training epoch."""
        self.current_epoch = int(epoch)

    def unfreeze_encoder(self, num_stages: int = 0) -> None:
        """Set the number of frozen encoder stages."""
        if not hasattr(self.encoder, "set_frozen_stages"):
            raise AttributeError("The encoder does not implement set_frozen_stages().")
        self.encoder.set_frozen_stages(num_stages)

    def get_optimizer_groups(self, lr: float, weight_decay: float) -> List[Dict[str, Any]]:
        """Return module-wise optimizer parameter groups."""
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        prototype_params = [p for p in self.prototype_lib.parameters() if p.requires_grad]

        other_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            lower_name = name.lower()
            if "encoder" in lower_name or "proto" in lower_name or "prototype" in lower_name:
                continue
            other_params.append(param)

        return [
            {"params": encoder_params, "lr": lr * 0.1, "weight_decay": weight_decay},
            {"params": prototype_params, "lr": lr * 2.0, "weight_decay": 0.0},
            {"params": other_params, "lr": lr, "weight_decay": weight_decay},
        ]

    @property
    def _module_device(self) -> torch.device:
        """Return the device of the module parameters."""
        return next(self.parameters()).device

    @staticmethod
    def _ensure_mask_shape(mask: torch.Tensor) -> torch.Tensor:
        """Ensure mask/logit tensor has shape [B, 1, H, W]."""
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4:
            raise ValueError(f"Expected tensor with shape [B, 1, H, W] or [B, H, W], got {tuple(mask.shape)}")
        return mask

    @staticmethod
    def _is_finite_tensor(tensor: Optional[torch.Tensor]) -> bool:
        """Return whether a tensor is finite."""
        if tensor is None:
            return False
        return bool(torch.isfinite(tensor).all().item())

    @staticmethod
    def _safe_float(value: Any) -> float:
        """Convert scalar-like values to Python float for logging."""
        if isinstance(value, torch.Tensor):
            return float(value.detach().item())
        return float(value)

