# Implementation Plan — SatQuery AI (3-Model Architecture)

- **System:** SatQuery AI (SIH 2026 PS 26167, ISRO)
- **Status:** v1.0 — Implementation roadmap for EarthDial + DOFA + DeltaVLM stack
- **Inputs:** PRD.md, Model_Selection.md, PS 26167
- **Date:** 2026-09-08

---

## 1. Implementation Principles

- **IP-1 Vertical slices:** End-to-end thin slice (GUI → controller → one tool → model → evidence) before deepening any layer.
- **IP-2 Real models from Day 1:** Base weights in registry; adapted checkpoints swap in via version bump.
- **IP-3 Contracts first:** API_Contracts.md frozen before producer/consumer implementation.
- **IP-4 Training external, non-blocking:** Fine-tuning runs on Colab in parallel; serving stack built against base weights.
- **IP-5 Windows dev, cloud deploy:** Local `.venv/Scripts/python.exe`; Docker optional for dev; cloud GPU for training/inference.
- **IP-6 Three models, two LLMs:** EarthDial (Phi-3-mini) for S1/S2/S4; Qwen3.5-2B for S3. DOFA frozen encoder.

---

## 2. Development Strategy

**Shape: Slice-first → Phase deepening → Hardening**

1. **Slice 1 (M1, ~1.5 weeks):** Minimal web page → backend → controller (VQA only) → registry → EarthDial base → confidence → evidence → trace.
2. **Phase 1–2 (M2):** Input infrastructure + all 5 specialist tools on base weights.
3. **Phase 3 (parallel from Week 1):** Data acquisition + fine-tuning (EarthDial on BEN.txt+VRSBench; DOFA head on BEN.txt pairs; DeltaVLM+Qwen on ChangeChat-105k).
4. **Phase 4–5 (M3):** Registry formalization + full agent (all 5 tasks, planner, aggregation, joint-use checks).
5. **Phase 6–8 (M4):** Full API, GUI, evidence/report engines — contract-complete.
6. **Phase 9–10 (M5):** Integration hardening + evaluation harness with dry-run measurement.
7. **Phase 11 (M6/M7):** Demo set, packaging, compliance bundle, readiness gates.

---

## 3. Repository Setup

Per Architecture (to be written):

```
satquery/
├── app/
│   ├── api/           # FastAPI routes, request/response models
│   ├── agent/         # Controller, task classifier, planner, aggregator
│   ├── registry/      # Tool registry, model store, descriptor schema
│   ├── tools/         # Specialist tool adapters (rs-vqa, rs-caption, rs-ground, change-vqa, opt-sar-fusion)
│   ├── runtime/       # Model loading, quantization, device management, load/evict
│   ├── evidence/      # Overlay renderer, evidence packaging
│   ├── reports/       # Report generation (HTML/PDF)
│   ├── events/        # Event stream, trace projection
│   └── storage/       # Session sandbox, cache, artifact store
├── config/
│   ├── app_config.yaml
│   ├── registry.json
│   └── sensor_profiles.yaml
├── frontend/          # React/Vue web app (TBD)
├── eval/              # Evaluation harness, benchmark runners
├── tests/             # Contract, suite, unit tests
├── scripts/           # Gates, acquisition, utilities
├── training/          # Fine-tuning scripts (Colab-targeted)
├── data/              # gitignored: raw/, processed/
├── models/            # gitignored: base weights, adapted checkpoints
├── sessions/          # gitignored: user sessions, traces
└── docs/              # This specification suite
```

---

## 4. Phase 0 — Project Foundation (IMP-001..IMP-010)

**Objective:** Running skeleton with config, logging/event-stream, storage sandbox, test infra.

| Task | Description | Requirements | Exit Criteria |
|------|-------------|--------------|---------------|
| IMP-001 | Create repo tree, .gitignore, README | Agent.md §6 | Tree exists; `scripts/gates.py` runs |
| IMP-002 | Config system: Pydantic settings, YAML loading, env overrides | C-10, C-12 | `config/app_config.yaml` loads; version echoed |
| IMP-003 | Event stream: structured JSONL writer + TraceView projection | TRD §14, §21 | Synthetic stream → well-formed TraceView |
| IMP-004 | Storage sandbox: path confinement, session isolation | AD-08, NFR-107 | Escaping path blocked; session isolation verified |
| IMP-005 | Pytest layout: markers (contract, suite, unit), fixtures | Agent.md §11 | `pytest -m contract` runs |
| IMP-006 | Local gate script: `scripts/gates.py` (import lint + test suite) | Agent.md §6/§18 | Green on Windows venv |
| IMP-007 | Logging: structlog, correlation IDs, trace integration | TRD §14 | Logs correlate with event stream |
| IMP-008 | Requirements: `requirements-inference.txt`, `requirements-training.txt` | C-12 | Both install cleanly in `.venv` |
| IMP-009 | Sensor profiles stub: Sentinel-2, Sentinel-1, Cartosat-2S, RISAT placeholders | OTD-103 | Profiles load; wavelength bands defined |
| IMP-010 | Decision Log initialized (append-only) | Agent.md §20 | `docs/Decision_Log.md` exists with header |

**Tests:** Event-stream round-trip, sandbox violation, config version-echo, smoke boot.
**Deliverables:** Repo skeleton, pinned requirements, config schema, registry/plan/profile stubs, gate script.

---

## 5. Phase 1 — Data & Input Infrastructure (IMP-011..IMP-018)

**Objective:** Upload-time ingestion + plan-time preprocessing for all supported input configs.

| Task | Description | Requirements | Exit Criteria |
|------|-------------|--------------|---------------|
| IMP-011 | GeoTIFF/TIFF reader: rasterio, metadata envelope (CRS, transform, bands, dtype, nodata) | FR-001, DAT-001 | ImageMetadataEnvelope produced (Contract §5) |
| IMP-012 | PNG/JPEG reader: benchmark designation gate (reject undesignated) | FR-016, DAT-003 | T-08/T-09 pass (rejection/acceptance) |
| IMP-013 | Modality detector: optical/MS/SAR from band count, metadata, filename heuristics | FR-008, AD-09 | Modality classified; fallback logged |
| IMP-014 | Pair validator: co-registration check (CRS, transform, bounds tolerance), temporal order | FR-011, AD-08/11 | T-10 (corrupt), T-13 (incompatible) pass |
| IMP-015 | Preprocessing profiles: band composites (RGB, false-color), SAR dB/log scaling, normalization stats | DAT-004/005/006 | Profiles for sentinel-2, sentinel-1, benchmark-rgb, cartosat-2s, risat |
| IMP-016 | Resize/tiling: dynamic tiling (EarthDial 512, DOFA 224, DeltaVLM 224) with coordinate mapping | DAT-007, AD-10 | Coordinate round-trip test passes |
| IMP-017 | Tensor cache: keyed by (session, image_hash, profile, tile_params), LRU eviction | DAT-008 | Cache hit/miss verified; memory bounded |
| IMP-018 | Input API: `/ingest` (single/pair), `/validate`, `/describe` endpoints | API_Contracts §6 | Contract tests (R-1..R-15) pass |

**Tests:** T-08..T-13, unit tests for metadata, modality fallback, pair tolerance, temporal order, cache keying, coordinate round-trip.
**Deliverables:** Ingestion module, ImageMetadataEnvelope, sensor profiles v1, preprocessing stages, pair-check, tensor cache, input API.

---

## 6. Phase 2 — Specialist Tools on Base Weights (IMP-019..IMP-030)

**Objective:** All 5 specialist tools independently callable via Contract §10–§15 on base weights.

### 6.1 Model Runtime (IMP-019, IMP-020)
- **IMP-019:** Runtime core: device manager (CUDA/MPS/CPU), quantization (BNB int8/int4, GGUF), memory budget tracker.
- **IMP-020:** Model store: versioned checkpoints, lineage metadata (base, adapter, dataset, gate scores), checksum verification.

### 6.2 Model A — EarthDial-4B Adapters (IMP-021..IMP-024)
- **IMP-021:** Acquire EarthDial-4B weights (RGB/MS variants); verify license; place in `models/earthdial/`.
- **IMP-022:** EarthDial adapter: chat template, multi-image input (band-fusion), generate + grounding parse.
- **IMP-023:** `rs-vqa` tool: single-image VQA (optical/MS/SAR) — instruction format per EarthDial.
- **IMP-024:** `rs-caption` tool: single-image captioning (same backbone, caption prompt).

### 6.3 Model B — DOFA Encoder + Fusion Head (IMP-025..IMP-027)
- **IMP-025:** Acquire DOFA ViT-B weights (HF earthflow/DOFA or TorchGeo); verify CC-BY-4.0.
- **IMP-026:** DOFA encoder wrapper: wavelength input → patch embed → frozen ViT forward → multi-modal embeddings.
- **IMP-027:** `opt-sar-fusion` tool: co-registered pair → DOFA embeddings (optical wavelengths + SAR wavelengths) → cross-attention fusion head → EarthDial LLM decode (shared Phi-3-mini via EarthDial adapter).

### 6.4 Model C — DeltaVLM + Qwen3.5-2B (IMP-028..IMP-029)
- **IMP-028:** Acquire Qwen3.5-2B (Apache-2.0); implement DeltaVLM Bi-VE (EVA-ViT-g/14) + IDPM (CSRM + Q-former) architecture; load Qwen3.5-2B as frozen decoder.
- **IMP-029:** `change-vqa` tool: bi-temporal pair → Bi-VE → IDPM → Qwen3.5-2B generate.

### 6.5 Optional Grounding (IMP-030)
- **IMP-030:** `rs-ground` tool (registry `enabled: false` initially): EarthDial native box output → image-space mapping → `not_located` honesty gate.

**Per-Tool Contract Tests (each):**
- Slot validation (R-5) before model load
- Parameter closure (R-6)
- Output tuple conformance (R-10): `{text, boxes?, confidence, evidence_ptr}`
- Empty output → `MDL_EMPTY_OUTPUT`
- Joint-use enforcement for `change-vqa`/`opt-sar-fusion` (AML-003) → `MDL_JOINT_USE_VIOLATION`
- Confidence extraction or honest `unavailable` (R-9)
- Load/evict + VRAM budget (R-02)

**Deliverables:** 5 tool modules + adapters, runtime with resource classes + load/evict/int8, uniform output normalizers, logprob confidence extraction.

---

## 7. Phase 3 — Model Adaptation / Fine-Tuning (IMP-031..IMP-038)

**Objective:** Produce 3 adapted checkpoints with full lineage + gate evidence (E-6 discipline).

| Task | Description | Data | Gate |
|------|-------------|------|------|
| IMP-031 | Dataset acquisition: BigEarthNet.txt (S1+S2 pairs), VRSBench train, ChangeChat-105k train, CDVQA train | — | Checksums verified; manifests/acquisitions.json |
| IMP-032 | Derived datasets: EarthDial instruction mix (BEN.txt captions/VQA/RED + VRSBench train), DOFA head pairs (BEN.txt S1+S2 captions/VQA), DeltaVLM+Qwen (ChangeChat-105k train) | PRD §9, Model_Selection §3 | Split hygiene: zero overlap with prescribed test splits (audit) |
| IMP-033 | FT-A: EarthDial LoRA (r=16–32) on LLM + projector; freeze ViT; multi-sensor band-fusion | BEN.txt + VRSBench train | BEN.txt bench split: binary VQA ≥70%, caption BLEU-4 ≥30; VRSBench holdout BLEU-1 ≥45 |
| IMP-034 | FT-B: DOFA fusion head (MLP + cross-attn) on BEN.txt co-registered pairs | BEN.txt paired annotations | S4 joint caption/VQA quality vs Model A solo (ablation) |
| IMP-035 | FT-C: DeltaVLM+Qwen LoRA on Qwen3.5-2B (r=16–32); selective FT Bi-VE last 2 blocks; FT Q-former | ChangeChat-105k train (87,935) | ChangeChat val: caption CIDEr ≥ baseline; binary Acc ≥90%; open QA review |
| IMP-036 | Checkpoint import: version bump in registry, lineage pins, checksums in `models/` | — | Registry shows v2 for each adapted model |
| IMP-037 | Dry-run measurement: adapted Model A on VRSBench test + RSVQA test; Model C on CDVQA test1 + test2 | Prescribed test splits | Logs saved; **no tuning** (DEC-021) |
| IMP-038 | Re-run Phase 2 tool tests on adapted weights → all green | — | No regressions |

**Training Environment:** Colab A100 40 GB (recommended); T4 16 GB fallback (bf16 LoRA, batch 1–2, grad checkpoint).
**Scripts:** `training/model_a/`, `training/model_b/`, `training/model_c/` with per-run READMEs.

---

## 8. Phase 4 — Tool Abstraction Layer / Registry (IMP-039..IMP-043)

**Objective:** Formalize registry as sole invocation route with declarative, versioned descriptors.

| Task | Description | Requirements |
|------|-------------|--------------|
| IMP-039 | Registry schema: descriptor (name, version, modality, count, resource_class, load_group, params, enabled) | AGT-003/004/006, INT-003 |
| IMP-040 | `config/registry.json` v1: all 5 entries + resource classes (earthdial, dofa, deltavlm) + load groups | NFR-104/105, PRD-CON-007 |
| IMP-041 | Registry lookup: controller → registry → runtime → tool adapter (no direct imports) | Contract §9, R-7 |
| IMP-042 | Stub-specialist extensibility proof: add dummy tool with zero controller changes | NFR-105 |
| IMP-043 | Registry↔model-store startup consistency check | — |

**Tests:** R-7 (registry closure), AC-AGT-003 structural inspection, stub-specialist green, `enabled: false` unselectable, version in trace.

---

## 9. Phase 5 — Agentic Controller (IMP-044..IMP-052)

**Objective:** Full agent: classify → validate → select → plan → execute → aggregate → evidence → trace.

| Task | Description | Requirements |
|------|-------------|--------------|
| IMP-044 | Task classifier: deterministic rules + keyword/embedding matcher over 5 labels (vqa, caption, grounding, change_vqa, opt_sar_fusion) | AGT-001, TRD §8 |
| IMP-045 | Input compatibility checker: modality/count/format vs tool schema | AGT-002, FR-016 |
| IMP-046 | Tool selector: registry query → permitted params → execution plan (sequential/parallel) | AGT-003/006 |
| IMP-047 | Planner templates: single-tool, dual-tool (fusion), conditional (grounding fallback) | AGT-004 |
| IMP-048 | Executor: registry invocation, timeout, retry, error mapping (VAL_*, MDL_*) | AGT-005, Contract §21 |
| IMP-049 | Output aggregator: text merge, box union, confidence combination (min/weighted-mean), evidence pointer collection | FR-012, M-10 |
| IMP-050 | Evidence renderer: boxes/heatmaps → overlays on original imagery (tiling-aware) | FR-012, TRD §7 |
| IMP-051 | Trace emitter: full execution summary (task, models, params, outputs, confidence, evidence, latency) | FR-005, PRD §7 |
| IMP-052 | Controller API: `/query` (image(s) + text) → `{answer, evidence, trace, report_url}` | API_Contracts §7 |

**Tests:** Suite cases T-01..T-07 (end-to-end per task type), T-14 (multi-tool), T-15 (fallback), T-16 (trace completeness).

---

## 10. Phase 6 — Full API Surface (IMP-053..IMP-057)

| Task | Description |
|------|-------------|
| IMP-053 | `/health`, `/registry`, `/models`, `/sessions` endpoints |
| IMP-054 | `/ingest`, `/validate`, `/describe` (Phase 1) — harden |
| IMP-055 | `/query` (Phase 5) — harden: streaming optional, timeouts, cancellation |
| IMP-056 | `/evidence/{id}`, `/report/{id}`, `/trace/{id}` download endpoints |
| IMP-057 | OpenAPI spec generation; contract test suite (R-1..R-15) all green |

---

## 11. Phase 7 — GUI / Web Application (IMP-058..IMP-063)

| Task | Description |
|------|-------------|
| IMP-058 | Frontend scaffold: React + TypeScript + Vite (or Vue), Tailwind/Shadcn |
| IMP-059 | Upload zone: drag-drop GeoTIFF/PNG/JPEG; pair linking (co-reg / bi-temporal) |
| IMP-060 | Query panel: natural language input; task hint chips; example queries from PS |
| IMP-061 | Results panel: answer text, confidence badge, evidence overlays (leaflet/maplibre), trace accordion |
| IMP-062 | Report download: HTML/PDF with images, overlays, trace, confidence |
| IMP-063 | Session history: list, replay, export |

**Decision (DEC-027):** Frontend family TBD — spike in Week 1.

---

## 12. Phase 8 — Evidence & Reports (IMP-064..IMP-068)

| Task | Description |
|------|-------------|
| IMP-064 | Overlay renderer: boxes (xyxy), heatmaps, grid (DeltaVLM 3×3) → GeoTIFF/PNG |
| IMP-065 | Multi-modality evidence: per-modality overlays (optical, SAR) for S4 |
| IMP-066 | Report engine: Jinja2 HTML → WeasyPrint PDF; includes query, images, overlays, trace, confidence |
| IMP-067 | Benchmark-format adapters: VRSBench caption/box, RSVQA answer string, CDVQA answer vocab | OTD-111 |
| IMP-068 | Evaluation batch runner: same controller, batched per specialist, metric-ready outputs + traces | INT-002 |

---

## 13. Phase 9 — Integration Hardening (IMP-069..IMP-073)

| Task | Description |
|------|-------------|
| IMP-069 | Full suite green: all contract + suite + unit tests |
| IMP-070 | Load test: concurrent queries, load/evict under memory pressure |
| IMP-071 | Error injection: corrupt images, OOM, model load failure → graceful degradation |
| IMP-072 | Leakage firewall audit: no train data in test paths; split hygiene re-verified |
| IMP-073 | Determinism audit: greedy decoding, fixed seeds, reproducible traces (NFR-103) |

---

## 14. Phase 10 — Evaluation Harness & Dry-Run (IMP-074..IMP-078)

| Task | Description |
|------|-------------|
| IMP-074 | VRSBench test runner: captioning (BLEU/METEOR/ROUGE/CIDEr/CHAIR), grounding (Acc@IoU), VQA (GPT-judge or exact-match) |
| IMP-075 | RSVQA test runner: LR/HR overall + per-type accuracy |
| IMP-076 | CDVQA test runner: test1 + test2 OA/AA per question type |
| IMP-077 | Internal validation: BEN.txt benchmark split (Model A + Model B) |
| IMP-078 | Dry-run report: all prescribed benchmarks + internal; measurement logs (no tuning) |

---

## 15. Phase 11 — Demo, Packaging, Compliance (IMP-079..IMP-084)

| Task | Description |
|------|-------------|
| IMP-079 | Demo query set: 20+ representative queries covering all 5 tasks + edge cases |
| IMP-080 | Packaging: Dockerfile (cloud deploy), `models/` download scripts, `data/` acquisition scripts |
| IMP-081 | Compliance bundle: license audit (all weights/data), reproducibility manifest (checksums, versions, commands) |
| IMP-082 | Deliverable checklist: GUI + backend, codes, models, test demo, evaluation logs |
| IMP-083 | Readiness gates: all suite tests green; dry-run scores recorded; license audit clean |
| IMP-084 | Final Decision_Log append; Architecture.md / API_Contracts.md / TRD.md updated to match implementation |

---

## 16. Task Index (IMP-001..IMP-084)

| Phase | Tasks | Primary Components |
|-------|-------|-------------------|
| 0 | IMP-001..010 | Config, events, storage, tests, gates |
| 1 | IMP-011..018 | Ingestion, validation, preprocessing, cache, input API |
| 2 | IMP-019..030 | Runtime, 5 tool adapters (3 models) |
| 3 | IMP-031..038 | 3 fine-tunes + dry-runs |
| 4 | IMP-039..043 | Registry, descriptors, stub proof |
| 5 | IMP-044..052 | Classifier, selector, planner, executor, aggregator, evidence, trace, controller API |
| 6 | IMP-053..057 | Full REST API, OpenAPI, contract tests |
| 7 | IMP-058..063 | Web GUI (React/Vue) |
| 8 | IMP-064..068 | Evidence overlays, reports, benchmark adapters, batch runner |
| 9 | IMP-069..073 | Hardening, load, error injection, leakage audit, determinism |
| 10 | IMP-074..078 | Evaluation harness, prescribed benchmark dry-runs |
| 11 | IMP-079..084 | Demo set, packaging, compliance, deliverables |

---

## 17. Parallelizable Workstreams

| Stream | Tasks | Owner |
|--------|-------|-------|
| **Backend Core** | Phase 0, 1, 2 (runtime/tools), 4, 5, 6, 8, 9 | Main |
| **Model A Fine-tune** | IMP-031 (BEN.txt/VRSBench), IMP-033, IMP-036, IMP-037 | Colab |
| **Model B Fine-tune** | IMP-031 (BEN.txt pairs), IMP-034, IMP-036, IMP-037 | Colab |
| **Model C Fine-tune** | IMP-031 (ChangeChat-105k), IMP-035, IMP-036, IMP-037 | Colab |
| **Frontend** | IMP-058..063 (after Phase 5 API stable) | Parallel |
| **Evaluation** | IMP-074..078 (after Phase 3 checkpoints land) | Parallel |

---

## 18. Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| EarthDial weight license blocks Model A | S1/S2/S4 primary unusable | Fallback: InternVL3-1B + BEN.txt FT (Model_Selection.md primary) — ready as Plan B |
| DeltaVLM weights not on HF (reproduce Bi-VE+IDPM+Q-former) | S3 delay | Start reproduction early; Qwen3.5-2B + ChangeChat LoRA is fallback (Model_Selection.md backup) |
| DOFA wavelength inputs for Cartosat-2S/RISAT unknown | S4 sensor gap | Sensor profiles (IMP-009, OTD-103) + ISRO doc request; proxy test on Sentinel |
| 3 models × 2 LLMs exceed GPU budget | Deployment failure | Int8 co-residency (~8.5 GB) + lazy load/evict; validate on cloud GPU early (IMP-020 spike) |
| CDVQA test split ambiguity (test1 vs test2 vs both) | Evaluation mismatch | Pin from benchmark release (TBD-005); run both, report both |
| VRSBench/RSVQA format mismatches zero scores | Score loss | Format adapters (IMP-067) pinned from benchmark specs early |

---

## 19. Documentation Maintenance

- Every PR updating behavior → updates matching doc (Agent.md §19)
- Decision_Log.md append for consequential choices (Agent.md §20)
- This plan versioned in `docs/Implementation_Plan.md`; changes = new version + Decision_Log entry

---

## 20. Exit Criteria Summary

| Milestone | Criteria |
|-----------|----------|
| M1 (Slice 1) | GUI → controller → `rs-vqa` (EarthDial base) → answer + evidence + trace; all Phase 0–1 tests green |
| M2 (Tools) | All 5 tools callable on base weights; Phase 2 tests green; Model B/C cloud runtime spikes resolved |
| M3 (Adaptation) | 3 adapted checkpoints imported; gates passed; Phase 2 tests green on adapted weights |
| M4 (Full Stack) | Full API + GUI contract-complete; evidence/reports working; suite tests green |
| M5 (Hardened) | Load/error/leakage/determinism audits clean; dry-run scores on all prescribed benchmarks recorded |
| M6 (Delivery) | Demo set runs; Docker + download scripts; license audit clean; compliance bundle ready |

---