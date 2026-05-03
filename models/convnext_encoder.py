"""ConvNeXt V2 encoder for ReCAP-Seg.

The encoder wraps timm ConvNeXt V2 backbones and returns multi-scale features
for dense prediction together with a projected global feature for auxiliary
attribute/prototype alignment.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


LOGGER = logging.getLogger("ReCAP-Seg")


class ConvNeXtV2Encoder(nn.Module):
    """ConvNeXt V2 encoder with multi-scale feature outputs.

    Args:
        model_name: Short model key, e.g., ``convnextv2_base``.
        pretrained: Whether to load pretrained weights from timm.
        frozen_stages: Number of early stages to freeze. ``0`` means no stage is frozen.
        output_dim: Output dimension of the projected global feature.
        drop_path_rate: Drop-path rate used by the backbone.
        out_indices: Feature stages returned by timm.

    Returns from ``forward``:
        multi_scale_features: list of feature maps from shallow to deep stages.
        global_feature: projected global feature with shape ``[B, output_dim]``.
        deepest_feature: deepest feature map before global pooling.
    """

    MODEL_CONFIGS: Dict[str, Dict[str, object]] = {
        "convnextv2_tiny": {
            "timm_name": "convnextv2_tiny.fcmae_ft_in22k_in1k_384",
            "dims": [96, 192, 384, 768],
        },
        "convnextv2_small": {
            "timm_name": "convnextv2_small.fcmae_ft_in22k_in1k_384",
            "dims": [96, 192, 384, 768],
        },
        "convnextv2_base": {
            "timm_name": "convnextv2_base.fcmae_ft_in22k_in1k_384",
            "dims": [128, 256, 512, 1024],
        },
        "convnextv2_large": {
            "timm_name": "convnextv2_large.fcmae_ft_in22k_in1k_384",
            "dims": [192, 384, 768, 1536],
        },
    }

    def __init__(
        self,
        model_name: str = "convnextv2_base",
        pretrained: bool = True,
        frozen_stages: int = 0,
        output_dim: int = 512,
        drop_path_rate: float = 0.1,
        out_indices: Sequence[int] = (0, 1, 2, 3),
    ) -> None:
        super().__init__()

        if model_name not in self.MODEL_CONFIGS:
            available = ", ".join(self.MODEL_CONFIGS.keys())
            raise ValueError(f"Unknown encoder '{model_name}'. Available encoders: {available}")

        self.model_name = model_name
        self.output_dim = int(output_dim)
        self.frozen_stages = int(frozen_stages)
        self.out_indices = tuple(out_indices)

        config = self.MODEL_CONFIGS[model_name]
        timm_name = str(config["timm_name"])

        self.backbone = self._create_backbone(
            timm_name=timm_name,
            fallback_name=model_name,
            pretrained=pretrained,
            out_indices=self.out_indices,
            drop_path_rate=drop_path_rate,
        )

        self.feature_dims = self._infer_feature_dims(config)
        self.embed_dim = self.feature_dims[-1]

        self.global_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(self.embed_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )

        self._freeze_stages()

        LOGGER.info("ConvNeXtV2Encoder initialized")
        LOGGER.info("  Model        : %s", self.model_name)
        LOGGER.info("  Pretrained   : %s", pretrained)
        LOGGER.info("  Feature dims : %s", self.feature_dims)
        LOGGER.info("  Output dim   : %d", self.output_dim)
        LOGGER.info("  Frozen stages: %d", self.frozen_stages)

    @staticmethod
    def _create_backbone(
        timm_name: str,
        fallback_name: str,
        pretrained: bool,
        out_indices: Sequence[int],
        drop_path_rate: float,
    ) -> nn.Module:
        """Create a timm feature-extraction backbone."""
        try:
            return timm.create_model(
                timm_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices,
                drop_path_rate=drop_path_rate,
            )
        except Exception as exc:
            LOGGER.warning("Failed to create timm model '%s': %s", timm_name, exc)
            LOGGER.warning("Falling back to timm model '%s'.", fallback_name)
            return timm.create_model(
                fallback_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices,
                drop_path_rate=drop_path_rate,
            )

    def _infer_feature_dims(self, config: Dict[str, object]) -> List[int]:
        """Infer feature dimensions from timm metadata when available."""
        if hasattr(self.backbone, "feature_info"):
            try:
                channels = self.backbone.feature_info.channels()
                if channels:
                    return list(channels)
            except Exception:
                pass
        return list(config["dims"])

    def _freeze_stages(self) -> None:
        """Freeze early backbone stages."""
        if self.frozen_stages <= 0:
            return

        if hasattr(self.backbone, "stem"):
            for param in self.backbone.stem.parameters():
                param.requires_grad = False

        if hasattr(self.backbone, "stages"):
            num_stages = min(self.frozen_stages, len(self.backbone.stages))
            for stage_idx in range(num_stages):
                for param in self.backbone.stages[stage_idx].parameters():
                    param.requires_grad = False
                LOGGER.info("Frozen encoder stage %d", stage_idx)

    def unfreeze_all(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        LOGGER.info("All encoder stages are unfrozen")

    def set_frozen_stages(self, num_stages: int) -> None:
        """Update the number of frozen stages."""
        self.unfreeze_all()
        self.frozen_stages = int(num_stages)
        self._freeze_stages()

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return multi-scale visual features."""
        features = self.backbone(x)
        return list(features)

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        """Return projected global image features."""
        features = self.forward_features(x)
        return self.global_proj(features[-1])

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        """Encode an image batch."""
        features = self.forward_features(x)
        deepest = features[-1]
        global_feature = self.global_proj(deepest)

        return {
            "multi_scale_features": features,
            "global_feature": global_feature,
            "deepest_feature": deepest,
        }

    def get_feature_dims(self) -> List[int]:
        """Return channel dimensions of the multi-scale features."""
        return list(self.feature_dims)

    def get_output_dim(self) -> int:
        """Return the projected global feature dimension."""
        return self.output_dim


class ConvNeXtV2EncoderWithNeck(ConvNeXtV2Encoder):
    """ConvNeXt V2 encoder with a lightweight FPN neck."""

    def __init__(
        self,
        model_name: str = "convnextv2_base",
        pretrained: bool = True,
        frozen_stages: int = 0,
        output_dim: int = 512,
        neck_channels: int = 256,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            frozen_stages=frozen_stages,
            output_dim=output_dim,
            drop_path_rate=drop_path_rate,
        )

        self.neck_channels = int(neck_channels)
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(dim, self.neck_channels, kernel_size=1) for dim in self.feature_dims]
        )
        self.fpn_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.neck_channels, self.neck_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(self.neck_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in self.feature_dims
            ]
        )

        self.feature_dims = [self.neck_channels for _ in self.feature_dims]
        LOGGER.info("FPN neck channels: %d", self.neck_channels)

    def forward_features_fpn(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Return FPN features and raw backbone features."""
        raw_features = super().forward_features(x)
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, raw_features)]

        for idx in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[idx],
                size=laterals[idx - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            laterals[idx - 1] = laterals[idx - 1] + upsampled

        fpn_features = [conv(feat) for conv, feat in zip(self.fpn_convs, laterals)]
        return fpn_features, raw_features

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        """Encode an image batch and return FPN-enhanced features."""
        fpn_features, raw_features = self.forward_features_fpn(x)
        deepest = raw_features[-1]
        global_feature = self.global_proj(deepest)

        return {
            "multi_scale_features": fpn_features,
            "raw_features": raw_features,
            "global_feature": global_feature,
            "deepest_feature": deepest,
        }

    def get_neck_channels(self) -> int:
        """Return the FPN feature dimension."""
        return self.neck_channels
