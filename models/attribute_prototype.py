"""Attribute prototype bank for ReCAP-Seg.

The prototype bank maintains slot-wise, category-level learnable prototypes. Text
initialization is optional and used only at initialization time. During training
and inference, the model retrieves prototypes from image-derived lesion embeddings
without querying a language model.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LOGGER = logging.getLogger("ReCAP-Seg")


class AttributePrototypeLibrary(nn.Module):
    """Slot-wise clinical attribute prototype bank.

    Args:
        embed_dim: Prototype embedding dimension.
        attribute_config: Optional task-specific attribute schema. If omitted,
            the default polyp schema is used.
        text_encoder_name: SentenceTransformer model name used for optional text initialization.
        use_text_init: Whether to initialize prototypes from attribute descriptions.
        temperature: Initial retrieval temperature.
        learnable_temperature: Whether the logit scale is learnable.
        logit_scale_max: Upper bound for the exponential logit scale.
    """

    DEFAULT_ATTRIBUTE_CONFIG: Dict[str, Dict[str, Any]] = {
        "multiplicity": {
            "values": ["single", "multiple"],
            "weight": 0.8,
            "descriptions": {
                "single": "a single polyp lesion in the image",
                "multiple": "multiple polyp lesions in the image",
            },
        },
        "attachment_form": {
            "values": ["sessile_flat", "pedunculated", "unknown"],
            "weight": 1.2,
            "descriptions": {
                "sessile_flat": "a sessile or flat polyp attached broadly to the mucosa",
                "pedunculated": "a pedunculated polyp with a stalk connected to the mucosa",
                "unknown": "a polyp with unclear attachment form",
            },
            "label_mapping": {
                "sessile": "sessile_flat",
                "flat": "sessile_flat",
                "flat_base_broad": "sessile_flat",
                "slender_stalk": "pedunculated",
                "wide_stalk": "pedunculated",
                "stalk": "pedunculated",
            },
        },
        "shape": {
            "values": [
                "round",
                "oval",
                "lobulated",
                "irregular",
                "flat_low",
                "protuberant",
                "dome_shaped",
                "elongated",
                "nodular",
                "columnar",
                "hillock_like",
                "annular_depression",
                "unknown",
            ],
            "weight": 1.5,
            "descriptions": {
                "round": "a polyp with a round circular shape",
                "oval": "a polyp with an oval elliptical shape",
                "lobulated": "a polyp with a lobulated multi-lobed shape",
                "irregular": "a polyp with an irregular asymmetric shape",
                "flat_low": "a flat low-profile polyp barely elevated from the mucosa",
                "protuberant": "a protuberant polyp protruding from the mucosa",
                "dome_shaped": "a dome-shaped polyp with smooth rounded elevation",
                "elongated": "an elongated polyp with extended morphology",
                "nodular": "a nodular polyp with small rounded projections",
                "columnar": "a columnar polyp with tall cylindrical morphology",
                "hillock_like": "a hillock-like polyp resembling a small mound",
                "annular_depression": "a polyp with annular ring-shaped depression",
                "unknown": "a polyp with unclear shape",
            },
            "label_mapping": {
                "dome-shaped": "dome_shaped",
                "hillock-like": "hillock_like",
            },
        },
        "surface_texture": {
            "values": ["smooth", "rough", "granular_nodular", "lobulated_surface", "erosion_ulcer", "unknown"],
            "weight": 1.5,
            "descriptions": {
                "smooth": "a polyp with smooth uniform surface texture",
                "rough": "a polyp with rough uneven surface texture",
                "granular_nodular": "a polyp with granular or nodular surface texture",
                "lobulated_surface": "a polyp with lobulated bumpy surface texture",
                "erosion_ulcer": "a polyp with surface erosion or ulceration",
                "unknown": "a polyp with unclear surface texture",
            },
            "label_mapping": {
                "granular": "granular_nodular",
                "nodular": "granular_nodular",
                "lobulated": "lobulated_surface",
                "erosion": "erosion_ulcer",
                "ulcer": "erosion_ulcer",
            },
        },
        "boundary": {
            "values": ["sharp", "irregular_margin", "unknown"],
            "weight": 1.2,
            "descriptions": {
                "sharp": "a polyp with a sharp clear well-defined boundary",
                "irregular_margin": "a polyp with an irregular or unclear margin",
                "unknown": "a polyp with unclear boundary definition",
            },
            "label_mapping": {
                "clear": "sharp",
                "well_defined": "sharp",
                "blurred": "irregular_margin",
                "fuzzy": "irregular_margin",
            },
        },
        "base_stalk": {
            "values": ["flat_base_broad", "slender_stalk", "wide_stalk", "short_stalk", "narrow_based", "unknown"],
            "weight": 1.0,
            "descriptions": {
                "flat_base_broad": "a polyp with a flat broad base attachment",
                "slender_stalk": "a polyp with a slender thin stalk",
                "wide_stalk": "a polyp with a wide thick stalk",
                "short_stalk": "a polyp with a short stalk",
                "narrow_based": "a polyp with a narrow-based attachment",
                "unknown": "a polyp with unclear base or stalk morphology",
            },
            "label_mapping": {
                "narrow-based": "narrow_based",
                "sessile": "flat_base_broad",
                "pedunculated": "slender_stalk",
            },
        },
        "mucosal_activity": {
            "values": ["normal", "congestion_erythema", "hemorrhagic", "mucus_secretion", "unknown"],
            "weight": 1.0,
            "descriptions": {
                "normal": "a polyp with normal mucosal appearance",
                "congestion_erythema": "a polyp with mucosal congestion or erythema",
                "hemorrhagic": "a polyp with hemorrhagic bleeding appearance",
                "mucus_secretion": "a polyp with mucus secretion on the surface",
                "unknown": "a polyp with unclear mucosal activity",
            },
            "label_mapping": {
                "erythema": "congestion_erythema",
                "congestion": "congestion_erythema",
                "bleeding": "hemorrhagic",
                "mucus": "mucus_secretion",
            },
        },
    }

    INVALID_LABELS = {"unknown", "not detected", "invalid", "none", "null", ""}

    def __init__(
        self,
        embed_dim: int = 512,
        attribute_config: Optional[Dict[str, Dict[str, Any]]] = None,
        text_encoder_name: str = "all-MiniLM-L6-v2",
        use_text_init: bool = True,
        temperature: float = 0.1,
        learnable_temperature: bool = True,
        logit_scale_max: float = 100.0,
    ) -> None:
        super().__init__()

        self.embed_dim = int(embed_dim)
        self.attribute_config = copy.deepcopy(attribute_config or self.DEFAULT_ATTRIBUTE_CONFIG)
        self.temperature = float(temperature)
        self.logit_scale_max = float(logit_scale_max)

        initial_logit_scale = torch.log(torch.tensor(1.0 / self.temperature, dtype=torch.float32))
        if learnable_temperature:
            self.logit_scale = nn.Parameter(initial_logit_scale)
        else:
            self.register_buffer("logit_scale", initial_logit_scale)

        self.prototypes = nn.ParameterDict()
        self.value_to_idx: Dict[str, Dict[str, int]] = {}
        self.idx_to_value: Dict[str, Dict[int, str]] = {}

        self._validate_attribute_config()
        self._init_prototypes(text_encoder_name=text_encoder_name, use_text_init=use_text_init)
        self._build_index_mappings()

        LOGGER.info("AttributePrototypeLibrary initialized")
        LOGGER.info("  Slots           : %d", self.get_num_slots())
        LOGGER.info("  Total prototypes: %d", sum(self.get_num_values(slot) for slot in self.get_slot_names()))
        LOGGER.info("  Embed dim       : %d", self.embed_dim)
        LOGGER.info("  Text init       : %s", use_text_init)

    def _validate_attribute_config(self) -> None:
        """Validate attribute schema format."""
        if not isinstance(self.attribute_config, dict) or not self.attribute_config:
            raise ValueError("attribute_config must be a non-empty dictionary.")

        for slot, config in self.attribute_config.items():
            values = config.get("values")
            if not isinstance(values, list) or not values:
                raise ValueError(f"Slot '{slot}' must define a non-empty list of values.")
            if len(values) != len(set(values)):
                raise ValueError(f"Slot '{slot}' contains duplicate values: {values}")

    def _init_prototypes(self, text_encoder_name: str, use_text_init: bool) -> None:
        """Initialize learnable prototype vectors."""
        text_encoder = None
        if use_text_init:
            try:
                from sentence_transformers import SentenceTransformer

                LOGGER.info("Loading text encoder for prototype initialization: %s", text_encoder_name)
                text_encoder = SentenceTransformer(text_encoder_name)
            except Exception as exc:
                LOGGER.warning("Text initialization is disabled: %s", exc)
                text_encoder = None

        for slot, config in self.attribute_config.items():
            values = config["values"]
            if text_encoder is not None:
                init = self._encode_descriptions(text_encoder, slot, values, config.get("descriptions", {}))
            else:
                init = torch.randn(len(values), self.embed_dim)

            init = F.normalize(init.float(), dim=-1)
            self.prototypes[slot] = nn.Parameter(init)

    def _encode_descriptions(
        self,
        text_encoder: Any,
        slot: str,
        values: List[str],
        descriptions: Dict[str, str],
    ) -> torch.Tensor:
        """Encode textual descriptions and project them if needed."""
        texts = [descriptions.get(value, self._default_description(slot, value)) for value in values]
        embeddings = text_encoder.encode(texts, convert_to_tensor=True).detach().cpu().float()

        if embeddings.shape[-1] == self.embed_dim:
            return embeddings

        # A deterministic random projection avoids adding an unused learnable text
        # projection module that is not used after initialization.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
        projection = torch.randn(embeddings.shape[-1], self.embed_dim, generator=generator) / (embeddings.shape[-1] ** 0.5)
        return embeddings @ projection

    @staticmethod
    def _default_description(slot: str, value: str) -> str:
        """Fallback text description for a prototype value."""
        return f"a lesion with {slot.replace('_', ' ')} described as {value.replace('_', ' ')}"

    def _build_index_mappings(self) -> None:
        """Build label-to-index and index-to-label mappings."""
        for slot, config in self.attribute_config.items():
            values = config["values"]
            value_to_idx = {value: idx for idx, value in enumerate(values)}
            idx_to_value = {idx: value for idx, value in enumerate(values)}

            for raw_label, canonical_label in config.get("label_mapping", {}).items():
                if canonical_label in value_to_idx:
                    value_to_idx[raw_label] = value_to_idx[canonical_label]

            self.value_to_idx[slot] = value_to_idx
            self.idx_to_value[slot] = idx_to_value

    def get_prototypes(self, slot: str, normalize: bool = True) -> torch.Tensor:
        """Return prototypes for one slot."""
        self._check_slot(slot)
        prototypes = self.prototypes[slot]
        return F.normalize(prototypes, dim=-1) if normalize else prototypes

    def get_all_prototypes(self, normalize: bool = True) -> Dict[str, torch.Tensor]:
        """Return prototypes for all slots."""
        return {slot: self.get_prototypes(slot, normalize=normalize) for slot in self.get_slot_names()}

    def compute_similarity(
        self,
        embeddings: torch.Tensor,
        slot: str,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Compute scaled cosine similarity between embeddings and slot prototypes."""
        self._check_slot(slot)
        if normalize:
            embeddings = F.normalize(embeddings, dim=-1)
        prototypes = self.get_prototypes(slot, normalize=True)
        scale = self.logit_scale.exp().clamp(max=self.logit_scale_max)
        return scale * embeddings @ prototypes.t()

    def retrieve_soft(self, embeddings: torch.Tensor, slot: str) -> torch.Tensor:
        """Retrieve a probability-weighted prototype vector for one slot."""
        similarity = self.compute_similarity(embeddings, slot)
        weights = F.softmax(similarity, dim=-1)
        prototypes = self.get_prototypes(slot, normalize=True)
        return weights @ prototypes

    def retrieve_hard(self, embeddings: torch.Tensor, slot: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve the nearest prototype vector and its index for one slot."""
        similarity = self.compute_similarity(embeddings, slot)
        indices = similarity.argmax(dim=-1)
        prototypes = self.get_prototypes(slot, normalize=True)
        return prototypes[indices], indices

    def retrieve_all_slots(self, embeddings: torch.Tensor, soft: bool = True) -> torch.Tensor:
        """Retrieve attribute embeddings for all slots."""
        slot_embeddings = []
        for slot in self.get_slot_names():
            if soft:
                slot_embedding = self.retrieve_soft(embeddings, slot)
            else:
                slot_embedding, _ = self.retrieve_hard(embeddings, slot)
            slot_embeddings.append(slot_embedding)
        return torch.stack(slot_embeddings, dim=1)

    def build_attribute_embedding(
        self,
        labels: Optional[Dict[str, List[str]]],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Build slot-wise attribute embeddings from one sample's structured labels."""
        if device is None:
            device = next(self.parameters()).device
        labels = labels or {}

        slot_embeddings = []
        for slot in self.get_slot_names():
            prototypes = self.get_prototypes(slot, normalize=True)
            indices = self.labels_to_indices(slot, labels.get(slot, []))
            if indices:
                slot_embedding = prototypes[indices].mean(dim=0)
            else:
                slot_embedding = prototypes.mean(dim=0)
            slot_embeddings.append(slot_embedding)

        return torch.stack(slot_embeddings, dim=0).to(device)

    def build_batch_attribute_embeddings(
        self,
        labels_list: List[Optional[Dict[str, List[str]]]],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Build slot-wise attribute embeddings for a batch."""
        if device is None:
            device = next(self.parameters()).device
        embeddings = [self.build_attribute_embedding(labels, device=device) for labels in labels_list]
        output = torch.stack(embeddings, dim=0)

        if not torch.isfinite(output).all():
            LOGGER.warning("Non-finite attribute embeddings detected; using default embeddings.")
            default_embedding = self.get_default_embedding(device=device)
            output = default_embedding.unsqueeze(0).expand(len(labels_list), -1, -1).clone()

        return output.clamp(min=-10.0, max=10.0)

    def get_default_embedding(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Return per-slot average prototypes."""
        if device is None:
            device = next(self.parameters()).device
        slot_embeddings = [self.get_prototypes(slot, normalize=True).mean(dim=0) for slot in self.get_slot_names()]
        return torch.stack(slot_embeddings, dim=0).to(device)

    # Backward-compatible internal name used by older model code.
    def _get_default_embedding(self) -> torch.Tensor:
        return self.get_default_embedding()

    def predict_attributes(self, embeddings: torch.Tensor) -> Dict[str, Dict[str, Any]]:
        """Predict attribute categories using prototype similarity."""
        predictions: Dict[str, Dict[str, Any]] = {}
        for slot in self.get_slot_names():
            similarity = self.compute_similarity(embeddings, slot)
            probs = F.softmax(similarity, dim=-1)
            pred_idx = probs.argmax(dim=-1)
            pred_values = [self.idx_to_label(slot, int(idx.item())) for idx in pred_idx]
            predictions[slot] = {
                "probs": probs,
                "pred_idx": pred_idx,
                "pred_values": pred_values,
            }
        return predictions

    def get_slot_weight(self, slot: str) -> float:
        """Return loss weight for one attribute slot."""
        self._check_slot(slot)
        return float(self.attribute_config[slot].get("weight", 1.0))

    def get_num_values(self, slot: str) -> int:
        """Return number of values in one slot."""
        self._check_slot(slot)
        return len(self.attribute_config[slot]["values"])

    def get_slot_names(self) -> List[str]:
        """Return all slot names."""
        return list(self.attribute_config.keys())

    def get_num_slots(self) -> int:
        """Return number of attribute slots."""
        return len(self.attribute_config)

    def idx_to_label(self, slot: str, idx: int) -> str:
        """Convert value index to label string."""
        self._check_slot(slot)
        if idx not in self.idx_to_value[slot]:
            raise IndexError(f"Invalid index {idx} for slot '{slot}'.")
        return self.idx_to_value[slot][idx]

    def index_to_label(self, slot: str, index: int) -> Optional[str]:
        """Backward-compatible index-to-label method."""
        self._check_slot(slot)
        return self.idx_to_value[slot].get(index)

    def label_to_idx(self, slot: str, label: str) -> int:
        """Convert label string to value index. Returns -1 for invalid labels."""
        self._check_slot(slot)
        return self.value_to_idx[slot].get(label, -1)

    def label_to_index(self, slot: str, label: str) -> Optional[int]:
        """Backward-compatible label-to-index method."""
        idx = self.label_to_idx(slot, label)
        return None if idx < 0 else idx

    def labels_to_indices(self, slot: str, labels: Iterable[str]) -> List[int]:
        """Convert possibly multi-label values into valid prototype indices."""
        self._check_slot(slot)
        indices: List[int] = []
        seen = set()
        for label in labels:
            normalized_label = str(label).strip()
            if normalized_label.lower() in self.INVALID_LABELS:
                continue
            idx = self.label_to_idx(slot, normalized_label)
            if idx >= 0 and idx not in seen:
                indices.append(idx)
                seen.add(idx)
        return indices

    def _check_slot(self, slot: str) -> None:
        """Validate that a slot exists."""
        if slot not in self.attribute_config:
            available = ", ".join(self.get_slot_names())
            raise KeyError(f"Unknown attribute slot '{slot}'. Available slots: {available}")

    def forward(self, embeddings: torch.Tensor, soft: bool = True) -> torch.Tensor:
        """Retrieve all slot-wise attribute embeddings."""
        return self.retrieve_all_slots(embeddings, soft=soft)
