# ReCAP-Seg: Prompt-Free Medical Image Segmentation via Retrievable Clinical Attribute Priors

ReCAP-Seg is a prompt-free medical image segmentation framework that leverages structured clinical attributes during training but performs image-only inference at deployment. It turns clinically grounded semantics (e.g., lesion shape/texture/boundary) into retrievable, slot-wise attribute priors, enabling automatic refinement without manual text prompts.

## Summary

Accurate medical image segmentation in practice requires not only pixel-level delineation but also **interpretable lesion semantics**. While vision–language methods can inject semantic knowledge, many rely on **explicit text prompts at test time**, which is impractical in routine clinical workflows. ReCAP-Seg addresses this by **distilling attribute supervision into a retrievable prototype space** and activating it via **image-driven retrieval** at inference.

**Key idea:**
- **Training:** use image–mask–attribute triplets to learn a lesion embedding aligned with **slot-wise attribute prototypes**.
- **Inference:** predict a coarse mask first, use it as a pseudo mask to compute a lesion embedding, retrieve **soft attribute priors**, and refine the final segmentation—**without any user-provided prompts**.

- ## Method

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


### Installation
```bash
conda create -n recapseg python=3.10 -y
conda activate recapseg

# Install PyTorch (match your CUDA)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
