# Requirements — SatQuery AI

- **System:** SatQuery AI (SIH 2026 PS 26167)
- **Status:** v1.0
- **Date:** 2026-09-08

---

## 1. Functional Requirements (FR)

| ID | Requirement | Trace |
|----|-------------|-------|
| FR-001 | Accept single optical/multispectral/SAR image (GeoTIFF/TIFF) | PS §Input Scope |
| FR-002 | Accept co-registered optical/MS + SAR pair (GeoTIFF/TIFF) | PS §Input Scope |
| FR-003 | Accept bi-temporal pair (GeoTIFF/TIFF) | PS §Input Scope |
| FR-004 | Accept PNG/JPEG only for designated benchmark datasets | PS §Input Scope |
| FR-005 | Single-image VQA on optical, multispectral, SAR | PS §Mandatory |
| FR-006 | Single-image captioning (mandatory S2 mode) | PS §Mandatory |
| FR-007 | Single-image text-guided grounding (optional S2 mode) | PS §Mandatory |
| FR-008 | Bi-temporal change description OR change-VQA (mandatory) | PS §Mandatory |
| FR-009 | Cross-modal optical–SAR joint analysis (mandatory) | PS §Mandatory |
| FR-010 | Agentic orchestration: classify → validate → select → execute → aggregate → evidence + trace | PS §Agentic |
| FR-011 | Remote-sensing adaptation: ≥1 component fine-tuned on BigEarthNet.txt or open data | PS §Mandatory |
| FR-012 | Evidence-grounded response: text + visual overlays + confidence + execution summary | PS §Expected |
| FR-013 | Interactive web GUI with upload, query, results, evidence, trace, report download | PS §Deliverables |
| FR-014 | Reject incompatible inputs with typed errors (VAL_*) before model invocation | Architecture §5 |
| FR-015 | Enforce joint-use for change-vqa and opt-sar-fusion (both images required) | Architecture §4.4/4.5 |

---

## 2. Non-Functional Requirements (NFR)

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-001 | Inference on single cloud GPU | ≤24 GB VRAM (int8 co-resident ~8.5 GB) |
| NFR-002 | Training on external Colab-class GPU | A100 40 GB |
| NFR-003 | Local development on Windows 11 | `.venv/Scripts/python.exe`, no Docker required |
| NFR-004 | Adapted weights redistributable | All licenses Apache-2.0 / MIT / CC-BY-4.0 |
| NFR-005 | Deterministic evaluation runs | Greedy decoding, fixed seeds, logged |
| NFR-006 | Full audit trace per query | Task, models, params, outputs, confidence, evidence, latency |
| NFR-007 | Leakage firewall | Zero train/test overlap; separate eval harness |
| NFR-008 | Contract-first APIs | OpenAPI spec; contract tests before consumers |
| NFR-009 | Registry-only invocation | No direct tool imports from controller |
| NFR-010 | Extensibility | Stub specialist added with zero controller changes |

---

## 3. Model-Specific Requirements

| Model | Requirement |
|-------|-------------|
| **EarthDial-4B (Model A)** | LoRA on BEN.txt + VRSBench train; supports optical/MS/SAR single-image VQA, caption, grounding |
| **DOFA ViT-B + Head (Model B)** | Frozen encoder; trainable fusion head on BEN.txt co-registered pairs; wavelength-conditioned input |
| **DeltaVLM + Qwen3.5-2B (Model C)** | Bi-VE selective FT + Q-former FT + Qwen LoRA on ChangeChat-105k; Qwen3.5-2B frozen (Apache-2.0) |

---

## 4. Benchmark Evaluation Requirements

| Benchmark | Task | Split | Metrics |
|-----------|------|-------|---------|
| VRSBench | Captioning, Grounding, VQA | Test (9,350 images) | BLEU/METEOR/ROUGE/CIDEr/CHAIR; Acc@IoU0.5/0.7; GPT-judge VQA |
| RSVQA-LR/HR | VQA | Official test | Overall + per-type accuracy |
| CDVQA | Change-VQA | Test1 + Test2 | OA, AA per 8 question types |
| BEN.txt | Internal validation | Benchmark split (1,082 pairs) | Binary VQA, MCQ, Caption, RED |
| ISRO/SAC | Hidden | Cartosat-2S + RISAT | Undisclosed (normalized) |

---

## 5. Deliverables Checklist

- [ ] Interactive web application (GUI + backend)
- [ ] All source code (backend, frontend, training scripts)
- [ ] Model weights / adapters (EarthDial LoRA, DOFA head, DeltaVLM+Qwen LoRA)
- [ ] Test demonstrations on prescribed benchmarks
- [ ] Reproducibility package: configs, lineage, checksums, evaluation logs
- [ ] License audit manifest