# Product Requirements Document - SatQuery AI

- **System:** SatQuery AI (SIH 2026 PS 26167, ISRO)
- **Status:** v1.0 - Initial PRD for 3-model architecture
- **Date:** 2026-09-08

---

## 1. Problem Statement (from PS 26167)

Develop an **interactive, agentic vision-language assistant** for analyzing single and paired remote-sensing images through natural-language queries.

- **Single-image baseline (mandatory):** VQA + captioning OR grounding on optical/multispectral/SAR
- **Principal focus:** Joint reasoning over cross-modal (optical+SAR) and bi-temporal pairs
- **Agentic orchestration:** Auto-select, sequence, execute specialist models per query + inputs
- **Evidence-grounded responses:** Text + visual overlays + confidence + execution trace

---

## 2. Input Scope (PS §Defined Input Scope)

| Config | Modalities | Tasks |
|--------|------------|-------|
| Single image | Optical / Multispectral / SAR | Captioning, VQA, Text-guided grounding |
| Cross-modal pair | Co-registered Optical/MS + SAR | Joint extraction, cross-modal analysis |
| Bi-temporal pair | Two spatially aligned images (diff. times) | Change detection, description, change-VQA |
| Formats | GeoTIFF/TIFF (primary); PNG/JPEG (benchmark-only) | - |

---

## 3. Mandatory Functional Scope (PS §Mandatory Functional Scope)

| ID | Requirement |
|----|-------------|
| FR-001 | Remote-sensing adaptation: ≥1 visual/VL component fine-tuned on BigEarthNet.txt or open-source data |
| FR-002 | Single-image VQA (mandatory) + captioning OR grounding |
| FR-003 | Bi-temporal change description OR change-VQA (mandatory) |
| FR-004 | Cross-modal optical–SAR joint analysis (mandatory) |
| FR-005 | Agentic controller: task classification → input validation → model/tool selection → execution → aggregation → evidence + trace |

---

## 4. Three-Model Architecture

| Model | Role | Capabilities Covered | Source |
|-------|------|---------------------|--------|
| **EarthDial-4B** (hiyamdebary/EarthDial) | S1/S2 Single-image specialist | Optical/MS/SAR single-image: captioning, VQA, grounding | https://github.com/hiyamdebary/EarthDial |
| **DOFA** (zhu-xlab/DOFA) | S4 Cross-modal fusion encoder | Co-registered optical/MS + SAR: feature extraction/fusion (encoder) | https://github.com/zhu-xlab/DOFA |
| **EarthDial-4B Multi-Modal** | S3 Bi-temporal & multi-sensor change | Bi-temporal pairs (Optical, SAR, MSI): change detection, description, change-VQA | Fine-tuned on BigEarthNet-MM |

> **Note on DOFA:** DOFA is a wavelength-conditioned ViT encoder (no LLM). It serves as the **feature extractor/fusion backbone** for S4. A lightweight projection + LLM head (or the EarthDial LLM) generates the final textual response for optical–SAR queries.

> **Note on Multi-Modal Change (Model C):** Replaces single-sensor optical baselines with EarthDial-4B multi-image engine fine-tuned on BigEarthNet (Sentinel-1 SAR dual-pol + Sentinel-2 MSI). This supports true cross-sensor and multi-temporal change detection under Apache-2.0 / open license.

---

## 5. Representative Queries (from PS)

- "Describe the land-cover and major objects visible in this image." → EarthDial
- "Highlight the water body referred to in the query." → EarthDial (grounding)
- "What changed between these two dates, and where did the change occur?" → EarthDial-4B Multi-Modal
- "Use the optical and SAR images together to identify built-up and water-covered regions." → DOFA + EarthDial LLM head
- "Has the built-up area increased, decreased, or remained unchanged?" → EarthDial-4B Multi-Modal

---

## 6. Deliverables

- Interactive web application (GUI + agentic backend)
- All code, model weights/adapters, training scripts
- Test demonstrations on prescribed benchmarks (VRSBench, RSVQA, CDVQA) + internal validation set
- Reproducibility package: configs, lineage, checksums, evaluation logs

---

## 7. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-001 | Inference on single cloud GPU (≤24 GB VRAM target; quantization allowed) |
| NFR-002 | Training on Colab-class external GPU (A100 40 GB) |
| NFR-003 | Local dev on Windows 11 (CPU/optional GPU); no Docker required for dev |
| NFR-004 | All adapted weights redistributable (license-compliant) |
| NFR-005 | Deterministic greedy decoding for evaluation runs |
| NFR-006 | Full audit trace: task → model → params → outputs → confidence → evidence |

---

## 8. Evaluation Targets (PS §Evaluation/Judging Criteria)

| Benchmark | Purpose | Split |
|-----------|---------|-------|
| VRSBench | Single-image captioning, grounding, VQA | Prescribed test split (9,350 images) |
| RSVQA-LR/HR | Single-image VQA | Official test splits |
| CDVQA | Bi-temporal change-VQA | Official test1 (39,686 QAs) + test2 (31,036 QAs) |
| ISRO/SAC | Hidden evaluation | Cartosat-2S optical + RISAT SAR pairs (undisclosed) |

Scores normalized across benchmarks before final ranking.

---

## 9. Constraints & Assumptions

- **DEC-001:** BigEarthNet.txt is primary adaptation dataset (co-registered S1+S2 + text)
- **DEC-002:** VRSBench/RSVQA/CDVQA train splits only for training; test splits held out
- **DEC-003:** Model C unified on EarthDial-4B Multi-Modal fine-tuned on BigEarthNet-MM to natively support Optical, SAR, and Multispectral change detection with zero 12h training timeout constraints
- **DEC-004:** DOFA used as frozen encoder; only projection head + optional LLM adapter trained
- **DEC-005:** EarthDial weights used as initialization; LoRA fine-tune on BEN.txt + VRSBench train
- **DEC-006:** Windows dev = storage/test only; cloud GPU = training + inference deployment