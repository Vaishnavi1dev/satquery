# Architecture — SatQuery AI (3-Model Architecture)

- **System:** SatQuery AI (SIH 2026 PS 26167, ISRO)
- **Status:** v1.0 — Architecture for EarthDial + DOFA + DeltaVLM stack
- **Date:** 2026-09-08

---

## 1. High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐   │
│  │   Upload    │  │   Query     │  │  Results    │  │   Trace /        │   │
│  │   Zone      │  │   Panel     │  │  Panel      │  │   Report         │   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬─────────┘   │
└─────────┼────────────────┼────────────────┼─────────────────┼──────────────┘
          │                │                │                 │
          ▼                ▼                ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API GATEWAY (FastAPI)                              │
│  /ingest  /validate  /describe  /query  /evidence  /report  /trace  /health │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGENTIC CONTROLLER                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Classifier  │  │  Validator   │  │   Selector   │  │   Planner    │    │
│  │  (5 tasks)   │  │  (modality,  │  │  (registry   │  │  (templates) │    │
│  │              │  │   count, fmt)│  │   query)     │  │              │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │                 │             │
│         ▼                 ▼                 ▼                 ▼             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     EXECUTOR / AGGREGATOR                             │   │
│  │  registry.invoke() → collect outputs → merge text/boxes → confidence │   │
│  │  evidence pointers → trace emission                                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
┌─────────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   MODEL A           │ │   MODEL B       │ │   MODEL C       │
│   EarthDial-4B      │ │   DOFA ViT-B    │ │   DeltaVLM +    │
│   (Phi-3-mini LLM)  │ │   (frozen enc)  │ │   Qwen3.5-2B    │
│   ┌───────────────┐ │ │   ┌───────────┐ │ │   ┌───────────┐ │
│   │ rs-vqa        │ │ │   │ Fusion    │ │ │   │ change-vqa│ │
│   │ rs-caption    │ │ │   │ Head      │ │ │   │           │ │
│   │ rs-ground(opt)│ │ │   │           │ │ │   │           │ │
│   └───────────────┘ │ │   └───────────┘ │ │   └───────────┘ │
└─────────┬───────────┘ └────────┬────────┘ └────────┬────────┘
          │                      │                   │
          └──────────────────────┼───────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODEL RUNTIME                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Device Mgr   │  │ Quantization │  │ Load/Evict   │  │ Model Store  │    │
│  │ (CUDA/CPU)   │  │ (BNB int8/4) │  │ (LRU, groups)│  │ (versioned)  │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TOOL REGISTRY (config/registry.json)                    │
│  Entries: rs-vqa, rs-caption, rs-ground, change-vqa, opt-sar-fusion         │
│  Each: name, version, modality, count, resource_class, load_group, params   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow — Query Execution

```
User Query + Image(s)
        │
        ▼
┌───────────────────┐
│  /ingest          │  →  ImageMetadataEnvelope(s) {modality, CRS, transform, bands, hash}
│  /validate        │  →  VAL_* errors or OK
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Classifier       │  →  TaskLabel ∈ {vqa, caption, grounding, change_vqa, opt_sar_fusion}
│  (deterministic)  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Validator        │  →  Checks: modality ✓, count ✓, format ✓, pair compatibility ✓
│  (input vs tool)  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Selector         │  →  Registry query → ExecutionPlan {steps: [{tool, params, depends_on}]}
│  (registry)       │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Executor         │  →  For each step: runtime.load(model) → tool.invoke(tensors, params)
│  (sequential/par) │       → {text, boxes?, confidence, evidence_ptr}
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Aggregator       │  →  Merge: text concatenation, box union, confidence combination
│                   │       evidence_ptr collection
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Evidence Render  │  →  Overlays on original imagery (tiling-aware coordinate mapping)
│                   │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Trace Emit       │  →  Structured event stream (JSONL) → TraceView projection
│                   │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Response         │  →  {answer, evidence_urls, trace_id, report_url, confidence}
└───────────────────┘
```

---

## 3. Model Runtime Architecture

### 3.1 Resource Classes
| Class | Models | VRAM (int8) | Load Group |
|-------|--------|-------------|------------|
| `earthdial` | EarthDial-4B (Phi-3-mini) | ~4.2 GB | `llm_primary` |
| `dofa` | DOFA ViT-B + Fusion Head | ~0.3 GB | `encoder` |
| `deltavlm` | DeltaVLM (Bi-VE + Q-former) + Qwen3.5-2B | ~4.0 GB | `llm_secondary` |

### 3.2 Load Groups & Eviction Policy
- **`llm_primary`** (EarthDial): Resident for S1/S2/S4 queries. Evicted only if `llm_secondary` needed and VRAM pressure.
- **`llm_secondary`** (Qwen3.5-2B): Loaded on-demand for S3. Evicted after query completes.
- **`encoder`** (DOFA): Always resident (negligible VRAM).

### 3.3 Quantization
- All LLMs: BNB int8 default (bitsandbytes). int4 fallback if VRAM < 8 GB.
- Encoders (DOFA, Bi-VE): bf16 (small, no quantization benefit).

---

## 4. Specialist Tool Adapters (Contract §10–§15)

### 4.1 `rs-vqa` — Single-Image VQA (Model A: EarthDial)
```python
Input:  {image_tensor: [C,H,W], query: str, modality: "optical|multispectral|sar"}
Params: {max_new_tokens: 256, temperature: 0.0, do_sample: false}
Output: {text: str, boxes: null, confidence: float, evidence_ptr: "analysed_image"}
```

### 4.2 `rs-caption` — Single-Image Captioning (Model A: EarthDial)
```python
Input:  {image_tensor: [C,H,W], modality: "optical|multispectral|sar"}
Params: {max_new_tokens: 128, temperature: 0.0}
Output: {text: str, boxes: null, confidence: float, evidence_ptr: "analysed_image"}
```

### 4.3 `rs-ground` — Text-Guided Grounding (Model A: EarthDial, optional)
```python
Input:  {image_tensor: [C,H,W], query: str, modality: "optical|multispectral|sar"}
Params: {max_new_tokens: 128, temperature: 0.0}
Output: {text: str, boxes: [[x1,y1,x2,y2], ...], confidence: float, evidence_ptr: "analysed_image"}
Honesty: If no boxes → {text: "not_located", boxes: [], confidence: 0.0, ...}
```

### 4.4 `change-vqa` — Bi-Temporal Change VQA (Model C: DeltaVLM+Qwen)
```python
Input:  {image_t1: [C,H,W], image_t2: [C,H,W], query: str}
Params: {max_new_tokens: 256, temperature: 0.0}
Output: {text: str, boxes: [[x1,y1,x2,y2], ...]?, confidence: float, evidence_ptr: "bi_temporal_pair"}
Joint-use: REQUIRES both images → MDL_JOINT_USE_VIOLATION if only one provided
```

### 4.5 `opt-sar-fusion` — Cross-Modal Optical–SAR Joint Analysis (Model B + Model A LLM)
```python
Input:  {optical_tensor: [C1,H,W], sar_tensor: [C2,H,W], query: str,
         optical_wavelengths: [µm], sar_wavelengths: [µm]}
Params: {max_new_tokens: 256, temperature: 0.0}
Pipeline:
  1. DOFA encoder: optical_tensor + optical_wavelengths → embed_opt [D]
  2. DOFA encoder: sar_tensor + sar_wavelengths → embed_sar [D]
  3. Fusion Head: cross_attn(embed_opt, embed_sar) → fused_embed [D]
  4. Projection: fused_embed → LLM embedding space
  5. EarthDial LLM (Phi-3-mini): decode → text
Output: {text: str, boxes: null, confidence: float, evidence_ptr: "opt_sar_pair"}
Joint-use: REQUIRES both modalities → MDL_JOINT_USE_VIOLATION if only one provided
```

---

## 5. Input Processing Pipeline

### 5.1 Sensor Profiles (`config/sensor_profiles.yaml`)
```yaml
sentinel-2:
  bands: [B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12]
  wavelengths_um: [0.49, 0.56, 0.665, 0.842, 0.705, 0.74, 0.783, 0.865, 1.61, 2.19]
  gsd_m: 10
  composites:
    rgb: [B04, B03, B02]
    false_color: [B08, B04, B03]
  normalization: {mean: [...], std: [...]}

sentinel-1:
  bands: [VV, VH]
  wavelengths_um: [0.056, 0.056]  # C-band ~5.6 cm
  gsd_m: 10
  scaling: "db"  # 10*log10(power)
  normalization: {mean: [...], std: [...]}

cartosat-2s:
  bands: [B, G, R, NIR]  # TBD from ISRO docs
  wavelengths_um: [TBD]
  gsd_m: TBD
  composites: {rgb: [R, G, B]}
  normalization: {mean: [...], std: [...]}

risat:
  bands: [HH, HV]  # TBD
  wavelengths_um: [TBD]  # C-band
  gsd_m: TBD
  scaling: "db"
  normalization: {mean: [...], std: [...]}

benchmark-rgb:
  bands: [R, G, B]
  wavelengths_um: [0.65, 0.55, 0.45]  # approximate
  gsd_m: "varies"
  composites: {rgb: [R, G, B]}
  normalization: "imagenet"
```

### 5.2 Preprocessing Stages
1. **Read** → GeoTIFF (rasterio) or PNG/JPEG (PIL) → numpy array [H,W,C]
2. **Modality Detect** → band count + metadata + filename heuristic → `modality`
3. **Profile Select** → sensor profile from metadata or user designation
4. **Band Composite** → select bands per profile.composites[rrgb|false_color] → [H,W,3]
5. **SAR Scaling** → if SAR: 10*log10(power + eps) → dB
6. **Normalize** → (x - mean) / std per profile
7. **Resize/Tile** → model-specific:
   - EarthDial: 512×512 (native)
   - DOFA: 224×224 (ViT patch)
   - DeltaVLM Bi-VE: 224×224
8. **Tensor Cache** → key = (session_id, image_hash, profile_name, tile_params)

---

## 6. Evidence & Reporting

### 6.1 Evidence Types
| Type | Source | Rendering |
|------|--------|-----------|
| `analysed_image` | Single-image tools | Original image + optional boxes/heatmap overlay |
| `bi_temporal_pair` | `change-vqa` | Side-by-side T1/T2 + change heatmap / 3×3 grid |
| `opt_sar_pair` | `opt-sar-fusion` | Optical + SAR side-by-side + fusion attention map |

### 6.2 Report Structure (HTML → PDF)
- **Header:** Query, timestamp, trace_id, session_id
- **Input:** Thumbnails, metadata envelopes, modality badges
- **Execution Trace:** Table of steps (tool, model, params, latency, output summary)
- **Answer:** Primary text response
- **Evidence:** Embedded overlays (base64 PNG or linked files)
- **Confidence:** Per-component + aggregated, with semantics label
- **Footer:** Model versions, registry version, reproducibility checksums

---

## 7. Storage & Session Management

### 7.1 Session Sandbox
```
sessions/
└── {session_id}/
    ├── images/
    │   ├── {image_hash}.tif          # original upload
    │   └── {image_hash}_{profile}.pt # cached tensor
    ├── traces/
    │   └── {trace_id}.jsonl          # event stream
    ├── evidence/
    │   └── {evidence_id}.png         # rendered overlays
    └── reports/
        └── {report_id}.html/.pdf
```
- Path confinement: all session paths rooted under `sessions/{session_id}/`
- LRU cleanup: max 100 sessions or 10 GB, configurable

---

## 8. Configuration Artifacts

| File | Purpose |
|------|---------|
| `config/app_config.yaml` | Runtime settings: device, VRAM budget, quantization, cache sizes, timeouts |
| `config/registry.json` | Tool descriptors (versioned), resource classes, load groups, permitted params |
| `config/sensor_profiles.yaml` | Band maps, wavelengths, composites, normalization per sensor |
| `config/model_profiles.yaml` | Model-specific: chat templates, input resolutions, tokenizer settings |

---

## 9. API Contracts Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness + model registry status |
| `/registry` | GET | Full registry descriptor list |
| `/ingest` | POST | Upload image(s) → {image_ids, metadata_envelopes} |
| `/validate` | POST | Check image(s) compatibility for a task → {valid, errors} |
| `/describe` | GET | Image metadata by ID |
| `/query` | POST | Main endpoint: {image_ids[], query} → {answer, evidence, trace, report} |
| `/evidence/{id}` | GET | Download evidence overlay |
| `/report/{id}` | GET | Download HTML/PDF report |
| `/trace/{id}` | GET | Download full trace JSONL |
| `/sessions` | GET/POST | Session management |

---

## 10. Cross-Cutting Concerns

| Concern | Mechanism |
|---------|-----------|
| **Determinism** | Greedy decoding (temp=0), fixed seeds, logged in trace |
| **Leakage Firewall** | Train/test split enforcement in data loaders; evaluation harness uses separate code paths |
| **Observability** | Structured JSONL event stream (correlation_id = trace_id); TraceView projection |
| **Error Taxonomy** | `VAL_*` (input), `MDL_*` (model), `SYS_*` (system) — mapped to HTTP codes |
| **License Compliance** | License manifest in `models/LICENSES.json`; verified at checkpoint import |
| **Reproducibility** | Checksums for all weights, datasets, code commits; recorded in trace |

---

## 11. Deployment Topology

### 11.1 Development (Windows 11)
- `.venv/Scripts/python.exe` — no Docker
- CPU or optional local GPU
- `data/`, `models/`, `sessions/` local (gitignored)

### 11.2 Cloud Inference (Linux GPU)
- Docker container (optional but recommended for deploy)
- Single GPU (e.g., A10G 24 GB, T4 16 GB, or L4 24 GB)
- Models loaded from `models/` (baked in image or mounted volume)
- FastAPI + Uvicorn (workers=1, async)

### 11.3 Training (Colab / Cloud GPU)
- Separate environment (`requirements-training.txt`)
- Scripts in `training/{model_a,model_b,model_c}/`
- Outputs: adapted checkpoints → `models/` (versioned) + lineage JSON

---

## 12. Integration Boundaries

| Boundary | Rule |
|----------|------|
| `app/api` → `app/agent` | Only via Controller contract (single `query()` method) |
| `app/agent` → `app/tools` | Only via Registry handles (no direct imports) |
| `app/tools` → `app/runtime` | Only via `runtime.load(model_key)` + `model.forward()` |
| `app/runtime` → `models/` | Only via Model Store (versioned, checksummed) |
| `training/` → `app/` | **Never** — training is external; checkpoints imported via Model Store |

---

## 13. Decision Log Pointers

| Decision | Document |
|----------|----------|
| DEC-001: BigEarthNet.txt primary adaptation dataset | PRD.md |
| DEC-002: Train splits only, test splits held out | PRD.md, Data_Pipeline.md |
| DEC-003: DeltaVLM LLM → Qwen3.5-2B (Apache-2.0) | Model_Selection.md §3.3, §6 |
| DEC-004: DOFA frozen encoder + trainable head | Model_Selection.md §3.2 |
| DEC-005: EarthDial LoRA on BEN.txt + VRSBench | Model_Selection.md §3.1 |
| DEC-006: Windows dev / cloud deploy split | PRD.md §9 |
| DEC-007: Three models, two LLMs (no single generic VLM) | Model_Selection.md §5 |