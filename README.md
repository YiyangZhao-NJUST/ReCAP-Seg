# ReCAP-Seg: Prompt-Free Medical Image Segmentation via Retrievable Clinical Attribute Priors

ReCAP-Seg is a prompt-free medical image segmentation framework that leverages structured clinical attributes during training but performs image-only inference at deployment. It turns clinically grounded semantics (e.g., lesion shape/texture/boundary) into retrievable, slot-wise attribute priors, enabling automatic refinement without manual text prompts.

## Summary

Accurate medical image segmentation in practice requires not only pixel-level delineation but also **interpretable lesion semantics**. While vision–language methods can inject semantic knowledge, many rely on **explicit text prompts at test time**, which is impractical in routine clinical workflows. ReCAP-Seg addresses this by **distilling attribute supervision into a retrievable prototype space** and activating it via **image-driven retrieval** at inference.

**Key idea:**
- **Training:** use image–mask–attribute triplets to learn a lesion embedding aligned with **slot-wise attribute prototypes**.
- **Inference:** predict a coarse mask first, use it as a pseudo mask to compute a lesion embedding, retrieve **soft attribute priors**, and refine the final segmentation—**without any user-provided prompts**.

## Method

### 1) Mask-guided Multi-Scale ROI Pooling (Lesion-Centric Embedding)
ReCAP-Seg constructs a **lesion-centric embedding** by pooling multi-scale encoder features **inside the lesion region** (mask-weighted average pooling), then fusing multi-scale ROI vectors via **concatenation + projection**.  
- **Training:** uses ground-truth masks  
- **Inference:** uses coarse predicted masks (pseudo masks) to reduce train–test mismatch

### 2) Slot-wise Attribute Prototype Library (Text-Initialized, Learnable)
A **prototype library** is maintained for a fixed set of clinical attribute slots:
- `multiplicity`, `attachment form`, `shape`, `surface texture`, `boundary`, `base stalk`, `mucosal activity`

For each slot, ReCAP-Seg learns one prototype per attribute category in a shared embedding space. Prototypes are **initialized from short textual descriptions** (sentence-level language model), then optimized end-to-end. Retrieval is done via **temperature-scaled cosine similarity** to obtain per-slot distributions and soft attribute embeddings.

### 3) Dual-Branch Decoder with Cross-Attention + Learnable Fusion
ReCAP-Seg uses:
- a **plain visual decoder** (stable image-only prediction)
- an **attribute-guided decoder** that injects retrieved slot-wise attribute tokens via **cross-attention** at selected decoding stages

The two outputs are fused at the logit level using a **learnable fusion weight**, improving robustness when retrieval is uncertain.

### 4) Coarse-to-Fine Prompt-Free Inference
1. Plain decoder predicts a **coarse mask**  
2. Coarse mask becomes a **pseudo mask** for ROI pooling  
3. Lesion embedding retrieves **slot-wise attribute priors** from the prototype library  
4. Attribute-guided decoder refines segmentation → **final mask**

## Prompt Template Design for Attribute Generation

To build structured textual attributes for each training sample, we use **ChatGPT-4o** with vision capabilities as an automatic attribute annotator.

**Workflow**
1. We send the **raw image** together with its corresponding **segmentation mask** to ChatGPT-4o.  
2. We use a **structured prompt** to define the model role and enforce a standardized output format.  

The mask provides an explicit **region-of-interest (ROI)** cue that localizes the target lesion and reduces ambiguity. To ensure that the generated attributes are **clinically relevant, consistent, and reproducible**, we design a prompt template with explicit constraints:

- **Role Definition:** instruct the model to act as an experienced clinical expert and respond in a report-like style with medically appropriate terminology.  
- **Instruction:** require the model to describe the lesion strictly based on the provided image–mask pair and follow a fixed attribute schema.  
- **Output Requirement:** constrain the response to a predefined set of fields; if an attribute cannot be reliably confirmed, output `not detected` to avoid speculation.

We intentionally format outputs as **concise attribute lists** so that the resulting text fits within the input-length constraints of the **text encoder**, enabling subsequent vision–language alignment. Although we illustrate the template with a brain MRI glioma example, it can be readily adapted to other modalities and tasks by redefining the task-specific attribute set.

**Prompt (MRI Example)**

- **Role Definition:** Please act as an experienced radiology expert with many years of clinical practice, and respond in the style of a clinical report.  
- **Instruction:** Based on the provided brain MRI image and its corresponding mask, provide a comprehensive description of the glioma, strictly adhering to the following six aspects and giving only accurate and concise answers:  
  1) Number of lesions  
  2) Lesion location  
  3) Lesion shape  
  4) Boundary characteristics  
  5) Relationship with surrounding tissue (e.g., infiltration/edema/mass effect)  
  6) Internal features (e.g., necrosis/cystic change/calcification/hemorrhage)  
- **Output Requirement:** Report only features visible on the image; if not observed, mark as `not detected`. Do not include unrelated speculation or assumptions. Use standardized, concise, and medically appropriate terminology. Restrict your response strictly to the six aspects listed above.

---

### Installation
```bash
conda create -n recapseg python=3.10 -y
conda activate recapseg

# Install PyTorch (match your CUDA)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```
## Data Preparation

ReCAP-Seg evaluates on:

- Polyp: Kvasir, CVC-ClinicDB, CVC-ColonDB, CVC-300, ETIS-LaribPolypDB

- Ultrasound: TN3K, TG3K

- MRI: BrainMRI (FLAIR sequence)

- CT: MosMedData+

The training set uses (image, mask, attribute) triplets.
The procedure for obtaining structured attribute labels is provided in this repository (see the attribute annotation utilities/scripts).

Recommended folder convention (example):
```
data/
  polyp/
    images/  masks/  attributes.json
  tn3k/
    images/  masks/
  brainmri/
    images/  masks/
  mosmed/
    images/  masks/
```
Training
```
python train.py --dataset polyp --config configs/polyp.yaml
```
Inference (Prompt-Free)
```
python infer.py --input path/to/image_or_folder --output outputs/
```

Outputs typically include:

- predicted segmentation mask

- (optional) slot-wise attribute probabilities / predicted attributes for interpretability
