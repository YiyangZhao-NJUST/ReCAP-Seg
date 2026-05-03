"""Cross-attention decoder for ReCAP-Seg.

This module implements the plain visual decoder, retrieval-conditioned guided
decoder, and logit-level fusion used by ReCAP-Seg.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LOGGER = logging.getLogger("ReCAP-Seg")


class DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CrossAttention(nn.Module):
    """Cross-attention from image features to attribute tokens.

    The image feature sequence is used as Query, while retrieved attribute tokens
    are used as Key and Value.
    """

    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if query_dim % num_heads != 0:
            raise ValueError(f"query_dim ({query_dim}) must be divisible by num_heads ({num_heads}).")

        self.query_dim = int(query_dim)
        self.kv_dim = int(kv_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.query_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.query_dim, self.query_dim)
        self.k_proj = nn.Linear(self.kv_dim, self.query_dim)
        self.v_proj = nn.Linear(self.kv_dim, self.query_dim)
        self.out_proj = nn.Linear(self.query_dim, self.query_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.query_dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply cross-attention.

        Args:
            query: Image feature sequence with shape ``[B, N, C]``.
            key_value: Attribute tokens with shape ``[B, S, D]``.
            return_attention: Whether to return attention weights.
        """
        if query.ndim != 3:
            raise ValueError(f"Expected query shape [B, N, C], got {tuple(query.shape)}")
        if key_value.ndim != 3:
            raise ValueError(f"Expected key_value shape [B, S, D], got {tuple(key_value.shape)}")
        if query.shape[0] != key_value.shape[0]:
            raise ValueError("query and key_value must have the same batch size.")

        batch_size, num_query_tokens, channels = query.shape
        num_attr_tokens = key_value.shape[1]

        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)

        q = q.view(batch_size, num_query_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_attr_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_attr_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention - attention.amax(dim=-1, keepdim=True)
        attention = F.softmax(attention.float(), dim=-1).to(q.dtype)
        attention = torch.nan_to_num(attention, nan=1.0 / max(num_attr_tokens, 1), posinf=0.0, neginf=0.0)
        attention = self.dropout(attention)

        output = attention @ v
        output = output.transpose(1, 2).contiguous().view(batch_size, num_query_tokens, channels)
        output = self.out_proj(output)
        output = self.norm(query + output)

        if return_attention:
            return output, attention
        return output, None


class CrossAttentionBlock(nn.Module):
    """Cross-attention block with a feed-forward network."""

    def __init__(
        self,
        feat_dim: int,
        attr_dim: int,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.cross_attn = CrossAttention(
            query_dim=feat_dim,
            kv_dim=attr_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        hidden_dim = int(feat_dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feat_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(feat_dim)

    def forward(self, features: torch.Tensor, attr_emb: torch.Tensor) -> torch.Tensor:
        """Enhance a feature map with attribute tokens.

        Args:
            features: Feature map with shape ``[B, C, H, W]``.
            attr_emb: Attribute tokens with shape ``[B, S, D]``.
        """
        if features.ndim != 4:
            raise ValueError(f"Expected features shape [B, C, H, W], got {tuple(features.shape)}")

        batch_size, channels, height, width = features.shape
        feature_seq = features.flatten(2).transpose(1, 2)

        enhanced, _ = self.cross_attn(feature_seq, attr_emb)
        if not torch.isfinite(enhanced).all():
            LOGGER.warning("Non-finite values detected after cross-attention; returning original features.")
            return features

        ffn_output = self.ffn(enhanced)
        if not torch.isfinite(ffn_output).all():
            LOGGER.warning("Non-finite values detected in decoder FFN; returning original features.")
            return features

        enhanced = self.ffn_norm(enhanced + ffn_output)
        if not torch.isfinite(enhanced).all():
            LOGGER.warning("Non-finite values detected after FFN normalization; returning original features.")
            return features

        return enhanced.transpose(1, 2).reshape(batch_size, channels, height, width)


class PlainDecoder(nn.Module):
    """U-Net-style visual decoder without attribute conditioning."""

    def __init__(
        self,
        feature_dims: List[int],
        out_channels: int = 1,
        out_size: Tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()
        self._validate_feature_dims(feature_dims)

        self.out_size = tuple(out_size)
        c1, c2, c3, c4 = feature_dims

        self.up4 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c3 + c3, c3)

        self.up3 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c2 + c2, c2)

        self.up2 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c1 + c1, c1)

        head_mid_channels = max(c1 // 2, 1)
        self.head = nn.Sequential(
            nn.Conv2d(c1, head_mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(head_mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_mid_channels, out_channels, kernel_size=1),
        )

    @staticmethod
    def _validate_feature_dims(feature_dims: List[int]) -> None:
        if len(feature_dims) != 4:
            raise ValueError(f"Expected four feature dimensions [C1, C2, C3, C4], got {feature_dims}")

    @staticmethod
    def _match_size(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != target.shape[-2:]:
            x = F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Decode multi-scale visual features into segmentation logits."""
        if len(features) != 4:
            raise ValueError(f"Expected four feature maps, got {len(features)}")

        f1, f2, f3, f4 = features

        x = self.up4(f4)
        x = self._match_size(x, f3)
        x = self.dec4(torch.cat([x, f3], dim=1))

        x = self.up3(x)
        x = self._match_size(x, f2)
        x = self.dec3(torch.cat([x, f2], dim=1))

        x = self.up2(x)
        x = self._match_size(x, f1)
        x = self.dec2(torch.cat([x, f1], dim=1))

        logits = self.head(x)
        return F.interpolate(logits, size=self.out_size, mode="bilinear", align_corners=False)


class GuidedDecoder(nn.Module):
    """U-Net-style decoder conditioned on retrieved attribute tokens."""

    def __init__(
        self,
        feature_dims: List[int],
        attr_dim: int = 512,
        num_heads: int = 8,
        out_channels: int = 1,
        out_size: Tuple[int, int] = (256, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        PlainDecoder._validate_feature_dims(feature_dims)

        self.out_size = tuple(out_size)
        c1, c2, c3, c4 = feature_dims

        self.attr_proj_4 = nn.Linear(attr_dim, c4)
        self.attr_proj_3 = nn.Linear(attr_dim, c3)
        self.attr_proj_2 = nn.Linear(attr_dim, c2)

        self.ca_block_4 = CrossAttentionBlock(c4, c4, num_heads=num_heads, dropout=dropout)
        self.ca_block_3 = CrossAttentionBlock(c3, c3, num_heads=num_heads, dropout=dropout)
        self.ca_block_2 = CrossAttentionBlock(c2, c2, num_heads=num_heads, dropout=dropout)

        self.up4 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c3 + c3, c3)

        self.up3 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c2 + c2, c2)

        self.up2 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c1 + c1, c1)

        head_mid_channels = max(c1 // 2, 1)
        self.head = nn.Sequential(
            nn.Conv2d(c1, head_mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(head_mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_mid_channels, out_channels, kernel_size=1),
        )

    @staticmethod
    def _match_size(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != target.shape[-2:]:
            x = F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, features: List[torch.Tensor], attr_emb: torch.Tensor) -> torch.Tensor:
        """Decode visual features with attribute conditioning."""
        if len(features) != 4:
            raise ValueError(f"Expected four feature maps, got {len(features)}")
        if attr_emb is None:
            raise ValueError("attr_emb must not be None for GuidedDecoder.")
        if attr_emb.ndim != 3:
            raise ValueError(f"Expected attr_emb shape [B, S, D], got {tuple(attr_emb.shape)}")

        f1, f2, f3, f4 = features

        attr_4 = self.attr_proj_4(attr_emb)
        attr_3 = self.attr_proj_3(attr_emb)
        attr_2 = self.attr_proj_2(attr_emb)

        f4 = self.ca_block_4(f4, attr_4)
        x = self.up4(f4)
        x = self._match_size(x, f3)

        f3 = self.ca_block_3(f3, attr_3)
        x = self.dec4(torch.cat([x, f3], dim=1))

        x = self.up3(x)
        x = self._match_size(x, f2)

        f2 = self.ca_block_2(f2, attr_2)
        x = self.dec3(torch.cat([x, f2], dim=1))

        x = self.up2(x)
        x = self._match_size(x, f1)
        x = self.dec2(torch.cat([x, f1], dim=1))

        logits = self.head(x)
        return F.interpolate(logits, size=self.out_size, mode="bilinear", align_corners=False)


class DualBranchDecoder(nn.Module):
    """Plain and retrieval-conditioned decoder with logit-level fusion.

    Args:
        feature_dims: Multi-scale feature dimensions ``[C1, C2, C3, C4]``.
        attr_dim: Attribute token dimension.
        num_heads: Number of heads in cross-attention blocks.
        out_channels: Segmentation output channels.
        out_size: Output segmentation size.
        fusion_type: One of ``learnable``, ``fixed``, or ``gate``.
    """

    def __init__(
        self,
        feature_dims: List[int],
        attr_dim: int = 512,
        num_heads: int = 8,
        out_channels: int = 1,
        out_size: Tuple[int, int] = (256, 256),
        fusion_type: str = "learnable",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if fusion_type not in {"learnable", "fixed", "gate"}:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        self.fusion_type = fusion_type
        self.plain_decoder = PlainDecoder(
            feature_dims=feature_dims,
            out_channels=out_channels,
            out_size=out_size,
        )
        self.guided_decoder = GuidedDecoder(
            feature_dims=feature_dims,
            attr_dim=attr_dim,
            num_heads=num_heads,
            out_channels=out_channels,
            out_size=out_size,
            dropout=dropout,
        )

        if fusion_type == "learnable":
            self.fusion_weight = nn.Parameter(torch.tensor(0.0))
            self.gate = None
        elif fusion_type == "gate":
            self.gate = nn.Sequential(
                nn.Conv2d(out_channels * 2, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1),
                nn.Sigmoid(),
            )
            self.register_buffer("fusion_weight", torch.tensor(0.5))
        else:
            self.gate = None
            self.register_buffer("fusion_weight", torch.tensor(0.5))

        LOGGER.info("DualBranchDecoder initialized")
        LOGGER.info("  Feature dims: %s", feature_dims)
        LOGGER.info("  Attr dim    : %d", attr_dim)
        LOGGER.info("  Fusion type : %s", fusion_type)

    def forward(
        self,
        features: List[torch.Tensor],
        attr_emb: Optional[torch.Tensor] = None,
        return_both: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run plain decoding and optional guided decoding with fusion."""
        plain_logits = self.forward_plain_only(features)
        output: Dict[str, torch.Tensor] = {"plain_logits": plain_logits}

        if attr_emb is None:
            output["logits"] = plain_logits
            return output

        guided_logits = self.forward_guided_only(features, attr_emb)
        output["guided_logits"] = guided_logits

        if self.fusion_type == "learnable":
            weight = torch.sigmoid(self.fusion_weight)
            logits = weight * guided_logits + (1.0 - weight) * plain_logits
        elif self.fusion_type == "gate":
            if self.gate is None:
                raise RuntimeError("Gate module is not initialized.")
            weight = self.gate(torch.cat([plain_logits, guided_logits], dim=1))
            logits = weight * guided_logits + (1.0 - weight) * plain_logits
        else:
            weight = self.fusion_weight.to(device=plain_logits.device, dtype=plain_logits.dtype)
            logits = weight * guided_logits + (1.0 - weight) * plain_logits

        output["logits"] = logits
        output["fusion_weight"] = weight

        return output

    def forward_plain_only(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Run only the plain visual decoder."""
        return self.plain_decoder(features)

    def forward_guided_only(self, features: List[torch.Tensor], attr_emb: torch.Tensor) -> torch.Tensor:
        """Run only the retrieval-conditioned guided decoder."""
        return self.guided_decoder(features, attr_emb)
