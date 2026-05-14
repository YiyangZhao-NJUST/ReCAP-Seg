# ReCAP-Seg: Prompt-free Medical Image Segmentation via Retrievable Clinical Attribute Priors

[![Python](https://img.shields.io/badge/Python-3.10-blue)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1-red)](#installation)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green)](#installation)
[![Task](https://img.shields.io/badge/Task-Medical%20Image%20Segmentation-orange)](#)

Official implementation of **ReCAP-Seg**, a prompt-free medical image segmentation framework that learns from image--mask--attribute triplets during training and performs **image-only inference** at test time.

ReCAP-Seg converts structured clinical attribute annotations into retrievable semantic priors. During inference, the model first predicts a coarse mask, uses it to retrieve slot-wise attribute priors from a learned prototype bank, and then refines the segmentation without requiring any user-provided text prompt, attribute label, bounding box, or point prompt.

> **Paper:** *Prompt-free Medical Image Segmentation via Retrievable Clinical Attribute Priors*
> **Authors:** Yiyang Zhao, Yi Zhou, Jingxiong Li, Yingna Li, Tao Zhou

---

## Overview

Medical image segmentation requires accurate pixel-level delineation. In clinical workflows, lesion description also involves structured morphological and appearance cues, such as shape, boundary, texture, echogenicity, edema, opacity pattern, or lesion-tissue interface. Existing vision-language segmentation methods often rely on explicit textual prompts at inference, which is inconvenient for routine deployment.

ReCAP-Seg addresses this gap by learning a structured, retrievable semantic prior space during training and reusing it automatically during inference.

<p align="center">
  <img src="Method.png" width="90%">
</p>

The pipeline consists of three core components:

1. **Mask-guided Scale-aware Aggregation (MSA)**
   Aggregates multi-scale encoder features within the lesion region to obtain a lesion-centric embedding. Ground-truth masks are used during training, while coarse pseudo masks are used during inference.

2. **Clinical Attribute-induced Prototype Bank (CAPB)**
   Maintains text-initialized, learnable category-wise prototypes for structured attribute slots. Given a lesion embedding, ReCAP-Seg retrieves soft slot-wise attribute priors via similarity matching.

3. **Retrieval-conditioned Refinement Decoder (RRD)**
   Injects retrieved attribute priors into the decoder via cross-attention and combines the refined prediction with the plain prediction using learnable logit-level fusion.

---

## Attribute Construction and Quality Control

Public segmentation datasets usually provide images and masks but do not include structured clinical attribute labels. We therefore construct training attributes offline.

### Offline annotation workflow

1. **Input:** public, de-identified image--mask pairs.
2. **First-stage annotation:** ChatGPT-4o with vision capability is used as an automatic attribute annotator.
3. **ROI cue:** the segmentation mask is provided as a region-of-interest cue to reduce ambiguity from surrounding tissue or background.
4. **Constrained output:** a fixed prompt template enforces slot-wise, concise, and standardized outputs.
5. **Parsing:** raw outputs are deterministically parsed, normalized, and mapped to predefined discrete categories.
6. **Invalid-slot handling:** uncertain, contradictory, or unparsable slots are marked invalid and excluded from the corresponding attribute-based training objectives rather than being forcibly assigned.
7. **Quality control:** a locally deployed Qwen2.5-VL-32B-Instruct model scores each annotation for consistency, completeness, and schema conformity on a normalized 0--1 scale.
8. **Manual review:** low-scoring samples are manually reviewed and retained only after verification or correction.
9. **Clinician-assisted check:** particularly ambiguous cases are further examined with assistance from two clinical experts.

### Important clarification

ChatGPT-4o is used only during offline preprocessing. It is **not** used during model training, inference, or deployment. After attributes are constructed, ReCAP-Seg is trained and evaluated as an image-only segmentation model at inference time.

---

## Data Preparation

### Datasets

ReCAP-Seg is evaluated on four medical segmentation scenarios:

* **Polyp / Colonoscopy:** Kvasir, CVC-ClinicDB, CVC-ColonDB, CVC-300, ETIS-LaribPolypDB
* **Brain Tumor / MRI:** BrainMRI FLAIR dataset
* **Thyroid Nodule / Ultrasound:** TN3K, TG3K
* **Lung Infection / CT:** MosMedData+

Please download each dataset from its official source and follow its license or data-use agreement. This repository does not redistribute datasets unless explicitly permitted.

### Recommended data format

```text
data/
├── polyp/
│   ├── train/
│   │   ├── images/
│   │   ├── masks/
│   │   └── attributes.json
│   └── test/
│       ├── Kvasir/
│       ├── CVC-ClinicDB/
│       ├── CVC-ColonDB/
│       ├── CVC-300/
│       └── ETIS/
├── brainmri/
│   ├── train/images/
│   ├── train/masks/
│   └── train/attributes.json
├── thyroid/
│   ├── TN3K/
│   └── TG3K/
└── mosmed/
    ├── images/
    ├── masks/
    └── attributes.json
```

A typical `attributes.json` file can be organized as:

```json
{
  {
      "filename": "sample_0001.png",
      "labels": {
        "attachment_form": [
          "pedunculated"
        ],
        "shape": [
          "oval"
        ],
        "surface_texture": [
          "smooth"
        ],
        "boundary": [
          "sharp"
        ],
        "base_stalk": [
          "slender_stalk"
        ],
        "mucosal_activity": [
          "normal"
        ]
      },
  {
      "filename": "sample_0002.png",
      "labels": {
        "attachment_form": [
          "pedunculated"
        ],
        "shape": [
          "irregular"
        ],
        "surface_texture": [
          "granular_nodular",
          "rough"
        ],
        "boundary": [
          "irregular_margin"
        ],
        "base_stalk": [
          "slender_stalk"
        ],
        "mucosal_activity": [
          "congestion_erythema"
        ]
      }
    }
}
```

---

## Training

Example: train ReCAP-Seg on the polyp setting.

```bash
python train.py \
  --config configs/polyp.yaml \
  --data_root data/processed/polyp \
  --save_dir checkpoints/recapseg_polyp
```

