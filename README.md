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

## Highlights

* **Prompt-free inference:** only the input image is required at test time.
* **Retrievable clinical attribute priors:** structured lesion attributes are encoded as slot-wise prototypes and retrieved automatically from image-derived lesion embeddings.
* **Training-only attribute supervision:** structured attributes are used to regularize representation learning and prototype-space alignment during training, but are not required during inference.
* **Modality-adapted schemas:** the framework is shared across tasks, while the semantic attribute schema is specialized for each modality.
* **Coarse-to-fine refinement:** a plain branch predicts a coarse mask, and a retrieval-conditioned refinement decoder injects retrieved priors for final prediction.
* **Reproducible attribute construction:** prompt templates, deterministic parsing rules, and quality-control scripts are provided to support reproducibility.

---

## Overview

Medical image segmentation requires accurate pixel-level delineation. In clinical workflows, lesion description also involves structured morphological and appearance cues, such as shape, boundary, texture, echogenicity, edema, opacity pattern, or lesion-tissue interface. Existing vision-language segmentation methods often rely on explicit textual prompts at inference, which is inconvenient for routine deployment.

ReCAP-Seg addresses this gap by learning a structured, retrievable semantic prior space during training and reusing it automatically during inference.

<p align="center">
  <img src="assets/framework.png" width="90%">
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

## Task-/Modality-specific Attribute Schemas

The framework is shared across modalities, but the attribute schema is modality-adapted. In the current study, we instantiate seven structured slots for each modality for implementation consistency; only the semantic contents of the slots are task-specific.

| Task / Modality             | Attribute Schema                                                                                                                       | Clinical Rationale                                                                                                              |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Polyp / Colonoscopy         | Multiplicity; Attachment Form; Shape; Surface Texture; Boundary; Base Stalk; Mucosal Activity                                          | Captures endoscopic morphology, lesion-mucosa interface, and local mucosal response used in descriptive assessment of polyps.   |
| Brain Tumor / MRI           | Lesion Distribution; Shape; Margin Definition; Internal Heterogeneity; Peritumoral Interface; Mass Effect / Edema; Intensity Pattern   | Reflects tumor morphology, boundary clarity, intralesional heterogeneity, and surrounding tissue response in MRI.               |
| Thyroid Nodule / Ultrasound | Lesion Localization; Shape / Orientation; Margin; Echogenicity; Internal Composition; Calcification; Posterior Acoustic Feature / Halo | Captures standard ultrasound descriptors, including echogenicity, composition, margin, and acoustic signs.                      |
| Lung Infection / CT         | Distribution / Laterality; Extent; Shape; Margin; Internal Opacity Pattern; Pleural Relation; Associated Signs                         | Describes infection burden and CT appearance, including distribution, opacity pattern, boundary property, and associated signs. |

---

## Repository Structure

A recommended repository layout is shown below. The exact file names may be adjusted according to the released implementation.

```text
ReCAP-Seg/
├── configs/
│   ├── polyp.yaml
│   ├── brainmri.yaml
│   ├── tn3k.yaml
│   └── mosmed.yaml
├── data/
│   ├── polyp/
│   ├── brainmri/
│   ├── thyroid/
│   └── mosmed/
├── datasets/
│   └── dataset loaders
├── models/
│   ├── msa.py
│   ├── capb.py
│   ├── rrd.py
│   └── recapseg.py
├── prompts/
│   ├── attribute_prompt_templates/
│   └── baseline_prompt_templates/
├── tools/
│   ├── parse_attributes.py
│   ├── qc_attributes.py
│   └── convert_prompts_for_baselines.py
├── train.py
├── infer.py
├── eval.py
├── requirements.txt
└── README.md
```

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

The concrete field names should match the modality-specific schema and the parser used in `tools/parse_attributes.py`.

### Preprocessing

Unless otherwise specified, images are resized to `256 x 256` to maintain a controlled cross-modal evaluation setting.

```bash
python tools/preprocess.py \
  --dataset polyp \
  --data_root data/polyp \
  --save_root data/processed/polyp \
  --image_size 256
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

Example: train on BrainMRI.

```bash
python train.py \
  --config configs/brainmri.yaml \
  --data_root data/processed/brainmri \
  --save_dir checkpoints/recapseg_brainmri
```

Example: train on TN3K.

```bash
python train.py \
  --config configs/tn3k.yaml \
  --data_root data/processed/tn3k \
  --save_dir checkpoints/recapseg_tn3k
```

---

## Inference

Inference uses only images.

```bash
python infer.py \
  --config configs/polyp.yaml \
  --checkpoint checkpoints/recapseg_polyp/best.pth \
  --input data/processed/polyp/test/Kvasir/images \
  --output outputs/polyp/Kvasir
```

The output folder may contain:

```text
outputs/
├── masks/                  # final segmentation masks
├── coarse_masks/           # plain-branch coarse masks, optional
├── refined_masks/          # guided-branch masks, optional
└── attributes.json          # slot-wise predicted attribute probabilities, optional
```

The optional attribute predictions are induced by image-prototype similarity and are provided for structured semantic inspection. They are not required as test-time inputs.

---

## Evaluation

```bash
python eval.py \
  --pred outputs/polyp/Kvasir/masks \
  --gt data/processed/polyp/test/Kvasir/masks \
  --metrics dice miou
```

For cross-dataset polyp evaluation:

```bash
for dataset in Kvasir CVC-ClinicDB CVC-ColonDB CVC-300 ETIS; do
  python eval.py \
    --pred outputs/polyp/${dataset}/masks \
    --gt data/processed/polyp/test/${dataset}/masks \
    --metrics dice miou
done
```
