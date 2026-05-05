"""
ROI pooling modules for ReCAP-Seg.

This file implements mask-guided ROI pooling modules used to extract
lesion-centric representations from feature maps. The mask can be a
ground-truth mask during training or a coarse pseudo mask during inference.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskROIPooling(nn.Module):
    """
    Mask-guided ROI pooling.

    This module pools features inside the mask region and suppresses
    background-dominated responses. If the mask area is too small, the
    module falls back to global pooling by replacing the invalid mask with
    an all-one mask.
    """

    def __init__(
        self,
        pool_type: str = "weighted_avg",
        min_area_ratio: float = 0.01,
        output_dim: Optional[int] = None,
        input_dim: Optional[int] = None,
    ) -> None:
        """
        Args:
            pool_type: Pooling type. Supported options are
                ``weighted_avg``, ``weighted_max``, and ``avg``.
            min_area_ratio: Minimum valid mask area ratio. Masks with an
                area smaller than this threshold are replaced by full-image
                masks to avoid unstable ROI pooling.
            output_dim: Optional output dimension after projection.
            input_dim: Optional input feature dimension. Required when
                ``output_dim`` is used for projection.
        """
        super().__init__()

        valid_pool_types = {"weighted_avg", "weighted_max", "avg"}
        if pool_type not in valid_pool_types:
            raise ValueError(
                f"Unsupported pool_type: {pool_type}. "
                f"Expected one of {sorted(valid_pool_types)}."
            )

        if min_area_ratio < 0:
            raise ValueError("min_area_ratio must be non-negative.")

        self.pool_type = pool_type
        self.min_area_ratio = min_area_ratio

        if output_dim is not None and input_dim is not None and output_dim != input_dim:
            self.proj = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        else:
            self.proj = None

    @staticmethod
    def _resize_mask(
        mask: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Resize a mask to match the feature-map resolution.

        Args:
            mask: Tensor with shape ``[B, H, W]`` or ``[B, 1, H, W]``.
            target_size: Target spatial size ``(H, W)``.

        Returns:
            Resized mask with shape ``[B, 1, H, W]``.
        """
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        if mask.dim() != 4 or mask.size(1) != 1:
            raise ValueError(
                "mask must have shape [B, H, W] or [B, 1, H, W], "
                f"but got {tuple(mask.shape)}."
            )

        if mask.shape[-2:] != target_size:
            mask = F.interpolate(
                mask.float(),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        return mask.float()

    def _check_mask_validity(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Replace invalid or near-empty masks with full-image masks.

        Args:
            mask: Tensor with shape ``[B, 1, H, W]``.

        Returns:
            Valid mask with shape ``[B, 1, H, W]``.
        """
        batch_size = mask.shape[0]
        height, width = mask.shape[-2:]
        total_area = height * width

        mask_areas = mask.sum(dim=(1, 2, 3))
        min_area = total_area * self.min_area_ratio

        invalid = mask_areas < min_area
        if invalid.any():
            full_mask = torch.ones_like(mask)
            mask = torch.where(invalid.view(batch_size, 1, 1, 1), full_mask, mask)

        return mask

    @staticmethod
    def weighted_avg_pool(
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform weighted average pooling inside the mask region.

        Args:
            features: Feature tensor with shape ``[B, C, H, W]``.
            mask: Mask tensor with shape ``[B, 1, H, W]``.

        Returns:
            Pooled feature tensor with shape ``[B, C]``.
        """
        weighted_sum = (features * mask).sum(dim=(2, 3))
        mask_sum = mask.sum(dim=(2, 3)).clamp(min=1e-6)
        return weighted_sum / mask_sum

    @staticmethod
    def weighted_max_pool(
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform weighted max pooling inside the mask region.

        Args:
            features: Feature tensor with shape ``[B, C, H, W]``.
            mask: Mask tensor with shape ``[B, 1, H, W]``.

        Returns:
            Pooled feature tensor with shape ``[B, C]``.
        """
        mask_expanded = mask.expand_as(features)

        masked_features = features.masked_fill(mask_expanded < 0.5, float("-inf"))
        pooled = masked_features.flatten(2).max(dim=-1).values

        invalid = torch.isinf(pooled)
        if invalid.any():
            global_avg = features.mean(dim=(2, 3))
            pooled = torch.where(invalid, global_avg, pooled)

        return pooled

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: Feature tensor with shape ``[B, C, H, W]``.
            mask: Mask tensor with shape ``[B, H, W]`` or ``[B, 1, H, W]``.

        Returns:
            Pooled feature tensor with shape ``[B, C]`` or
            ``[B, output_dim]`` when projection is enabled.
        """
        if features.dim() != 4:
            raise ValueError(
                f"features must have shape [B, C, H, W], but got {tuple(features.shape)}."
            )

        _, _, height, width = features.shape

        mask = self._resize_mask(mask, (height, width))
        mask = self._check_mask_validity(mask)

        if self.pool_type == "weighted_avg":
            pooled = self.weighted_avg_pool(features, mask)
        elif self.pool_type == "weighted_max":
            pooled = self.weighted_max_pool(features, mask)
        elif self.pool_type == "avg":
            pooled = features.mean(dim=(2, 3))
        else:
            raise ValueError(f"Unsupported pool_type: {self.pool_type}")

        if self.proj is not None:
            pooled = self.proj(pooled)

        return pooled


class MultiScaleMaskROIPooling(nn.Module):
    """
    Multi-scale mask-guided ROI pooling.

    This module extracts ROI-level features from multiple encoder stages
    and fuses them into a unified lesion embedding.
    """

    def __init__(
        self,
        feature_dims: List[int],
        output_dim: int = 512,
        pool_type: str = "weighted_avg",
        fusion_type: str = "concat_proj",
    ) -> None:
        """
        Args:
            feature_dims: Channel dimensions of multi-scale feature maps.
            output_dim: Output embedding dimension.
            pool_type: ROI pooling type used at each scale.
            fusion_type: Multi-scale fusion type. Supported options are
                ``concat_proj``, ``add``, and ``attention``.
        """
        super().__init__()

        if len(feature_dims) == 0:
            raise ValueError("feature_dims must contain at least one feature dimension.")

        valid_fusion_types = {"concat_proj", "add", "attention"}
        if fusion_type not in valid_fusion_types:
            raise ValueError(
                f"Unsupported fusion_type: {fusion_type}. "
                f"Expected one of {sorted(valid_fusion_types)}."
            )

        self.feature_dims = feature_dims
        self.output_dim = output_dim
        self.fusion_type = fusion_type

        self.roi_pools = nn.ModuleList(
            [MaskROIPooling(pool_type=pool_type) for _ in feature_dims]
        )

        if fusion_type == "concat_proj":
            total_dim = sum(feature_dims)
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(inplace=True),
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim),
            )

        elif fusion_type == "add":
            self.projs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(dim, output_dim),
                        nn.LayerNorm(output_dim),
                    )
                    for dim in feature_dims
                ]
            )

        elif fusion_type == "attention":
            self.projs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(dim, output_dim),
                        nn.LayerNorm(output_dim),
                    )
                    for dim in feature_dims
                ]
            )
            self.attention = nn.Sequential(
                nn.Linear(output_dim * len(feature_dims), len(feature_dims)),
                nn.Softmax(dim=-1),
            )

    def forward(
        self,
        features_list: List[torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features_list: List of multi-scale feature maps. Each tensor
                should have shape ``[B, C_i, H_i, W_i]``.
            mask: Mask tensor with shape ``[B, H, W]`` or ``[B, 1, H, W]``.

        Returns:
            Multi-scale lesion embedding with shape ``[B, output_dim]``.
        """
        if len(features_list) != len(self.feature_dims):
            raise ValueError(
                f"Expected {len(self.feature_dims)} feature maps, "
                f"but got {len(features_list)}."
            )

        pooled_list = []
        for roi_pool, feature in zip(self.roi_pools, features_list):
            pooled_list.append(roi_pool(feature, mask))

        if self.fusion_type == "concat_proj":
            output = self.fusion(torch.cat(pooled_list, dim=-1))

        elif self.fusion_type == "add":
            projected = [proj(pooled) for proj, pooled in zip(self.projs, pooled_list)]
            output = torch.stack(projected, dim=0).mean(dim=0)

        elif self.fusion_type == "attention":
            projected = [proj(pooled) for proj, pooled in zip(self.projs, pooled_list)]
            stacked = torch.stack(projected, dim=1)

            concat = torch.cat(projected, dim=-1)
            weights = self.attention(concat).unsqueeze(-1)

            output = (stacked * weights).sum(dim=1)

        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")

        return output


class AdaptiveROIPooling(nn.Module):
    """
    Adaptive ROI pooling.

    This module adaptively combines mask-guided ROI pooling and global
    average pooling. It can be used when the mask quality varies across
    samples.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
    ) -> None:
        """
        Args:
            input_dim: Input feature dimension.
            output_dim: Output embedding dimension.
        """
        super().__init__()

        self.roi_pool = MaskROIPooling(
            pool_type="weighted_avg",
            output_dim=output_dim,
            input_dim=input_dim,
        )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        self.fusion_weight = nn.Sequential(
            nn.Linear(output_dim * 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        mask_confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: Feature tensor with shape ``[B, C, H, W]``.
            mask: Mask tensor with shape ``[B, H, W]`` or ``[B, 1, H, W]``.
            mask_confidence: Optional confidence score for each mask, with
                shape ``[B]`` or ``[B, 1]``. If provided, it is used as the
                fusion weight for ROI features.

        Returns:
            Pooled feature tensor with shape ``[B, output_dim]``.
        """
        roi_feat = self.roi_pool(features, mask)
        global_feat = self.global_pool(features)

        if mask_confidence is not None:
            weight = mask_confidence.view(-1, 1).to(features.dtype)
            weight = weight.clamp(0.0, 1.0)
        else:
            weight = self.fusion_weight(torch.cat([roi_feat, global_feat], dim=-1))

        output = weight * roi_feat + (1.0 - weight) * global_feat
        return output


def _test_roi_pooling() -> None:
    """Run a minimal sanity check for ROI pooling modules."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 4
    mask = torch.zeros(batch_size, 1, 384, 384, device=device)
    mask[:, :, 100:200, 100:200] = 1.0

    features = torch.randn(batch_size, 512, 24, 24, device=device)

    roi_pool = MaskROIPooling(pool_type="weighted_avg").to(device)
    pooled = roi_pool(features, mask)
    assert pooled.shape == (batch_size, 512), pooled.shape

    feature_dims = [128, 256, 512, 1024]
    features_list = [
        torch.randn(batch_size, 128, 96, 96, device=device),
        torch.randn(batch_size, 256, 48, 48, device=device),
        torch.randn(batch_size, 512, 24, 24, device=device),
        torch.randn(batch_size, 1024, 12, 12, device=device),
    ]

    multi_roi_pool = MultiScaleMaskROIPooling(
        feature_dims=feature_dims,
        output_dim=512,
        fusion_type="concat_proj",
    ).to(device)

    pooled_multi = multi_roi_pool(features_list, mask)
    assert pooled_multi.shape == (batch_size, 512), pooled_multi.shape

    adaptive_roi_pool = AdaptiveROIPooling(
        input_dim=512,
        output_dim=512,
    ).to(device)

    pooled_adaptive = adaptive_roi_pool(features, mask)
    assert pooled_adaptive.shape == (batch_size, 512), pooled_adaptive.shape

    print("All ROI pooling sanity checks passed.")


if __name__ == "__main__":
    _test_roi_pooling()
