# Model Selection - SatQuery AI (3-Model Architecture)

- **System:** SatQuery AI (SIH 2026 PS 26167, ISRO)
- **Status:** v1.0 - 3-model stack selection
- **Date:** 2026-09-08

---

## 1. Capability Slots (from PRD §3 / PS Mandatory Scope)

| Slot | Requirement | Input Config | Benchmark |
|------|-------------|--------------|-----------|
| **S1** | Single-image VQA (optical, multispectral, SAR) | 1 image | VRSBench test, RSVQA test |
| **S2** | Single-image Captioning (mandatory) + Grounding (optional) | 1 image | VRSBench test (captioning + grounding) |
| **S3** | Bi-temporal Change Description / Change-VQA (mandatory) | 2 images (bi-temporal) | CDVQA test1 + test2 |
| **S4** | Cross-modal Optical–SAR Joint Analysis | 2 images (co-registered optical/MS + SAR) | ISRO/SAC hidden set; BEN.txt benchmark split |

---

## 2. Three-Model Assignment

| Model | Slots | Backbone | Adaptation Strategy |
|-------|-------|----------|---------------------|
| **Model A: EarthDial-4B** | S1, S2 | InternVL2-4B (Phi-3-mini LLM) + band-fusion module | LoRA on BigEarthNet.txt (S1/S2/S4 annotations) + VRSBench train (caption/grounding/VQA) |
| **Model B: DOFA (ViT-B/L) + Projection Head** | S4 (encoder) | DOFA ViT-B/ViT-L (wavelength-conditioned hypernetwork) | Frozen encoder; train lightweight projection + cross-modal fusion head on BEN.txt co-registered pairs |
| **Model C: DeltaVLM (Bi-VE + IDPM + Q-former) + Qwen3.5-2B** | S3 | EVA-ViT-g/14 (Bi-VE) + CSRM + Q-former + Qwen3.5-2B (Apache-2.0) | LoRA on ChangeChat-105k (train split); Bi-VE selective FT (last 2 blocks); Q-former FT; Qwen3.5-2B frozen |

---

## 3. Model Details

### 3.1 Model A - EarthDial-4B (S1, S2)

| Aspect | Detail |
|--------|--------|
| **Repo** | https://github.com/hiyamdebary/EarthDial |
| **Weights** | HF: akshaydudhane/EarthDial_4B_RGB, EarthDial_4B_MS, EarthDial_4B_Methane_UHI |
| **Base** | InternVL2-4B (Phi-3-mini 3.8B LLM, MIT) |
| **Modality** | RGB, Multispectral (NIR via band-fusion), SAR, Bi-temporal sequences |
| **License** | Code: MIT. **HF weight license: UNVERIFIED - must confirm before use** |
| **Training Data** | 11.11M instruction pairs across 44 downstream datasets |
| **Published Perf** | Outperforms generic/RS VLMs across modalities; e.g., SAR/methane 77.09% vs GPT-4o 40.93% |
| **Compute** | 4B bf16 ≈ 8.4 GB; int8 ≈ 4.2 GB + overhead. Fits single-GPU cloud budget in int8. |
| **Adaptation Recipe** | LoRA (r=16–32) on LLM + projector; freeze ViT. Data mix: BEN.txt train+val (captions, binary/MCQ VQA, RED) + VRSBench train. Multi-sensor input formatting per EarthDial's band-fusion module. |
| **Gate** | BEN.txt benchmark split: binary VQA ≥70%, caption BLEU-4 ≥30; VRSBench-train holdout caption BLEU-1 ≥45 |

### 3.2 Model B - DOFA Encoder + Fusion Head (S4)

| Aspect | Detail |
|--------|--------|
| **Repo** | https://github.com/zhu-xlab/DOFA |
| **Weights** | HF: earthflow/DOFA (ViT-B, ViT-L); also via TorchGeo |
| **Architecture** | ViT + wavelength-conditioned dynamic hypernetwork (generates patch-embed weights from band center wavelengths). Pretrained on 5 EO modalities via MAE + knowledge distillation. |
| **License** | CC-BY-4.0 (permissive) |
| **Modality** | Any spectral configuration (Sentinel-1/2, Landsat, NAIP, etc.) via wavelength input |
| **Published Perf** | Linear probing / partial FT SOTA on 12 EO tasks (classification, segmentation, detection). Strong cross-sensor generalization. |
| **Compute** | ViT-B ≈ 86M params (≈0.3 GB); ViT-L ≈ 304M (≈1.2 GB). Negligible VRAM. |
| **Role in Stack** | **Frozen feature extractor** for co-registered optical/MS + SAR pairs. Outputs fused multi-modal embeddings. |
| **Trainable Head** | Lightweight projection (MLP) + cross-attention fusion → text decoder (EarthDial LLM or separate small LLM). Trained on BEN.txt paired annotations (captions, VQA, RED on S1+S2 pairs). |
| **Gate** | BEN.txt benchmark split (S4 tasks): joint captioning/VQA quality vs Model A solo; fusion ablation. |

### 3.3 Model C - DeltaVLM + Qwen3.5-2B (S3)

| Aspect | Detail |
|--------|--------|
| **Repo** | https://github.com/hanlinwu/DeltaVLM |
| **Paper** | DeltaVLM: Interactive RS Image Change Analysis via Instruction-Guided Difference Perception (Remote Sensing 2026, arXiv:2507.22346) |
| **Original LLM** | Vicuna-7B (LLaMA-2 lineage) - **NON-COMMERCIAL LICENSE** |
| **Our LLM** | **Qwen3.5-2B (Apache-2.0)** - license-compliant swap |
| **Architecture** | 1) Bi-temporal Vision Encoder (Bi-VE): EVA-ViT-g/14, selective FT (last 2 blocks). 2) Instruction-guided Difference Perception Module (IDPM): CSRM + Q-former. 3) LLM decoder (frozen). |
| **Training Data** | ChangeChat-105k (105K instructions on LEVIR-CC bi-temporal pairs): 6 tasks - captioning, binary classification, quantification, localization, open QA, multi-turn dialogue. |
| **License (Ours)** | Code: Apache-2.0 (repo). Dataset: ChangeChat-105k (CC-BY-4.0 per HF). Qwen3.5-2B: Apache-2.0. **All compliant.** |
| **Compute** | EVA-ViT-g ≈ 1.2B (≈2.4 GB); Q-former ≈ 0.1B; Qwen3.5-2B ≈ 2B (≈4.8 GB bf16). Total bf16 ≈ 7.5 GB. Int8 ≈ 4 GB. Fits budget. |
| **Adaptation Recipe** | LoRA on Qwen3.5-2B (r=16–32, α=32); selective FT Bi-VE last 2 blocks; FT Q-former. Freeze Qwen3.5-2B. Data: ChangeChat-105k train (87,935 samples). |
| **Gate** | ChangeChat-105k val: captioning CIDEr ≥ baseline; binary classification Acc ≥90%; open QA quality review. **Plus dry-run on prescribed CDVQA test1/test2** (measurement, not tuning). |

---

## 4. Weighted Decision Matrix (per Model_Selection.md §7 framework)

| Criterion | Weight | EarthDial (S1/S2) | DOFA+Head (S4) | DeltaVLM+Qwen (S3) |
|-----------|--------|-------------------|----------------|---------------------|
| C1 PS Requirement Fit | 16 | 5 (covers VQA/caption/grounding all modalities) | 5 (joint optical-SAR by design) | 5 (purpose-built for change-VQA) |
| C2 RS Domain Fit | 10 | 5 (11M RS instructions, multi-sensor) | 5 (pretrained on 5 EO modalities) | 5 (ChangeChat-105k, LEVIR-CC) |
| C3 Benchmark Relevance | 14 | 3 (no VRSBench/RSVQA/CDVQA published) | 2 (no prescribed benchmark numbers) | 3 (ChangeChat bench only; CDVQA dry-run needed) |
| C4 Multimodal Capability | 8 | 5 (RGB/MS/SAR/temporal) | 5 (any spectral config via wavelengths) | 4 (bi-temporal optical only) |
| C5 Reproducibility | 7 | 4 (code public; weight license TBD) | 5 (code+weights public, TorchGeo) | 4 (code public; LLM swap needed) |
| C6 Open Weights | 8 | 4 (weights public; license TBD) | 5 (HF + TorchGeo) | 4 (weights public; Vicuna license issue) |
| C7 Licensing | 8 | **4 (weight license unverified)** | 5 (CC-BY-4.0) | **5 (after Qwen swap: all Apache-2.0/CC-BY)** |
| C8 Compute Cost | 8 | 3 (4B = 8.4 GB bf16; int8 OK) | 5 (ViT-B tiny; head negligible) | 4 (≈7.5 GB bf16; int8 OK) |
| C9 Inference Speed | 3 | 3 (4B slower) | 5 (encoder only) | 3 (Bi-VE + Q-former + 2B LLM) |
| C10 Integration Ease | 5 | 3 (custom InternVL2 fork) | 4 (TorchGeo / HF transformers) | 3 (custom architecture; Q-former) |
| C11 Fine-tune Feasibility | 8 | 4 (LoRA on Phi-3; band-fusion complexity) | 5 (frozen encoder; tiny head) | 4 (Bi-VE selective FT + Q-former + LoRA) |
| C12 Evidence Generation | 3 | 4 (native grounding boxes) | 2 (encoder only; needs decoder) | 3 (localization grid; no native boxes) |
| C13 Confidence Estimation | 2 | 3 (logits accessible) | 2 (encoder logits only) | 3 (LLM logits) |
| **TOTAL** | **100** | **387** | **403** | **377** |

> **Note:** DOFA+Head scores highest because S4 is a pure fusion task where DOFA's wavelength-conditioned design is uniquely strong, and the trainable head is tiny. EarthDial and DeltaVLM lose points on benchmark relevance (no prescribed numbers) and integration friction, but are the only viable options for their respective slots.

---

## 5. Consolidation: Three Models (Not Two, Not Four)

| Slot | Model | Registry Entry |
|------|-------|----------------|
| S1 VQA | Model A (EarthDial) | `rs-vqa` |
| S2 Captioning | Model A (EarthDial) | `rs-caption` |
| S2 Grounding (optional) | Model A (EarthDial) | `rs-ground` |
| S3 Change-VQA | Model C (DeltaVLM+Qwen) | `change-vqa` |
| S4 Optical–SAR Fusion | Model B (DOFA encoder) + Model A LLM head | `opt-sar-fusion` |

- **Two LLM backbones**: EarthDial's Phi-3-mini (for S1/S2/S4 text generation) + Qwen3.5-2B (for S3 change-VQA).
- **One shared vision encoder for S4**: DOFA (frozen) extracts features; EarthDial's LLM decodes.
- **No single generic VLM** - satisfies PRD-CON-008 / AML-002.

---

## 6. Licensing Risks & Mitigations

| Risk | Model | Status | Mitigation |
|------|-------|--------|------------|
| EarthDial HF weight license unknown | A | **Open** | Verify HF model card / contact authors before Phase 2. Fallback: InternVL3-1B + BEN.txt FT (Model_Selection.md primary). |
| DeltaVLM original LLM (Vicuna-7B) non-commercial | C | **Resolved** | Swap to Qwen3.5-2B (Apache-2.0). Retain Bi-VE + IDPM + Q-former architecture; retrain on ChangeChat-105k. |
| DOFA CC-BY-4.0 requires attribution | B | Compliant | Include attribution in deliverable. |
| ChangeChat-105k / LEVIR-CC imagery redistribution | C | **Open** | Ship code + instructions + dataset download scripts; not imagery. Document provenance. |

---

## 7. Compute & Serving Plan

| Model | Params | bf16 VRAM | int8 VRAM | Deployment |
|-------|--------|-----------|-----------|------------|
| EarthDial-4B (Model A) | 4B | ~8.4 GB | ~4.2 GB | Cloud GPU (int8 resident) |
| DOFA ViT-B + Head (Model B) | ~100M | ~0.5 GB | ~0.3 GB | Co-resident (negligible) |
| DeltaVLM+Qwen3.5-2B (Model C) | ~3.3B | ~7.5 GB | ~4 GB | Cloud GPU (int8 resident) |
| **Total (int8 co-resident)** | - | - | **~8.5 GB** | Fits ≤16 GB GPU with headroom |

**Serving strategy:** Lazy load/evict per query plan (TRD §12). Both LLMs never resident simultaneously in bf16; int8 co-residency fits small GPU budget.

---

## 8. Unresolved Questions (TBD)

1. **TBD-001:** EarthDial HF weight license confirmation (blocker for Model A primary).
2. **TBD-002:** DeltaVLM Bi-VE + Q-former weight availability on HF (currently only code + dataset; weights may need reproduction).
3. **TBD-003:** DOFA ViT-B vs ViT-L choice for S4 (trade-off: quality vs VRAM; ViT-B likely sufficient for frozen encoder).
4. **TBD-004:** Whether Model A (EarthDial) single-image VQA on SAR matches RSVQA-HR distribution (aerial RGB) - may need RSVQA-HR train in mix.
5. **TBD-005:** Exact CDVQA test split protocol (test1 vs test2 vs both) - pin from benchmark release.
6. **TBD-006:** Sensor profile for Cartosat-2S (optical) + RISAT (SAR) - needed for DOFA wavelength inputs and EarthDial band-fusion.

---

## 9. Final Recommended Stack

| Layer | Selection | License | Key Evidence |
|-------|-----------|---------|--------------|
| **S1/S2/S4 Text Gen (Model A)** | **EarthDial-4B** (LoRA on BEN.txt + VRSBench train) | MIT code; **verify weights** | 44 RS benchmarks; multi-sensor native; grounding native |
| **S4 Fusion Encoder (Model B)** | **DOFA ViT-B** (frozen) + MLP projection + cross-attn head (trained on BEN.txt pairs) | CC-BY-4.0 | Wavelength-conditioned; 5-modality pretrain; SOTA cross-sensor |
| **S3 Change-VQA (Model C)** | **DeltaVLM (Bi-VE + IDPM + Q-former) + Qwen3.5-2B** (LoRA on ChangeChat-105k) | Apache-2.0 (all) | SOTA on RSICA (ChangeChat); purpose-built for bi-temporal interactive VQA |
| **Controller** | Deterministic task classifier + registry router (no neural model) | n/a | TRD §8 determinism-first |

**If EarthDial weight license blocks:** Fallback to **InternVL3-1B + BEN.txt multi-sensor LoRA** (Model_Selection.md primary) - same slots, cleaner license, smaller (1B), but requires full reproduction of RS-InternVL recipe.