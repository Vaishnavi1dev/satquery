# Decision Log - SatQuery AI

Append-only record of consequential choices (Agent.md §20). Newest entries at the bottom.
Format: `DEC-NNN | date | decision | rationale | status`.

---

| ID | Date | Decision | Rationale | Status |
|----|------|----------|-----------|--------|
| DEC-033 | 2026-09-08 | Model C decoder = `Qwen/Qwen3.5-2B-Base` text-LM submodule (frozen + LoRA); `Qwen2.5-1.5B/3B-Instruct` (model_profiles.yaml pin) kept as automatic fallback | Model_Selection.md §3.3 requires "Qwen3.5-2B (Apache-2.0)"; the only Apache-2.0 Qwen3.5-2B weights on HF are VLM-family checkpoints (`Qwen3_5ForConditionalGeneration`) - its text backbone is the faithful realization of the doc's "frozen LLM". Profile pin predates the doc and was not resolved by it | Implemented in training/model_c/ (IMP-035); profile pin unchanged pending explicit doc update |
| DEC-034 | 2026-09-08 | IMP-035 gates measured on ChangeChat-105k **test** files (`test_binary` 1,929 samples; `test` caption, 5 refs/pair) instead of the doc's "val" split | `hlwu/changechat-105k` ships train + 6 test JSONs only; no official val split exists. Measurement only, no tuning (DEC-021); CDVQA test1/test2 dry-run remains in IMP-037 | Implemented in training/model_c/ (IMP-035) |
| DEC-035 | 2026-09-13 | Supersede DEC-033: Model C decoder is EarthDial-4B Multi-Modal (Phi-3-mini), not Qwen3.5-2B; no Qwen/DeltaVLM component is used | Actual implemented and demoed stack (EarthDial-4B Model A/C + DOFA Model B); DEC-033 never entered the runtime registry | config/registry.json, Model_Selection.md updated |
| DEC-036 | 2026-09-13 | Removed post-hoc temperature scaling from confidence aggregation; the system now reports raw model-reported confidence (single tool value, or arithmetic mean across specialists) plus an explicit divergence/uncertainty flag | The T=1.15 factor was arbitrary and never fitted on a calibration set, so any "calibrated confidence" claim was unsupported; reporting the raw model value with a low-confidence/conflict flag is honest and auditable | app/agent/aggregator.py, app/agent/controller.py, tests updated; calibration_method="none", calibration_temperature=None |
