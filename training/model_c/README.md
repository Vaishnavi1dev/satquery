# Model C — DeltaVLM + Qwen3.5-2B Fine-Tune (IMP-035)

**Per-run README for `training/model_c/`.** Implements **FT-C** from `docs/Implementation_Plan.md`
(IMP-035) per `docs/Model_Selection.md` §3.3 and `docs/Requirements.md` §3.

## Artifacts

| File | Purpose |
|------|---------|
| `train_deltavlm_qwen.ipynb` | Colab/Kaggle training notebook (the runnable deliverable) |
| `build_notebook.py` | Regenerates the notebook (`python training/model_c/build_notebook.py`) |

## What it does

Trains the S3 change-VQA stack on ChangeChat-105k train (87,935 samples):

- **Bi-VE** — EVA-ViT-g/14 (LAVIS `eva_vit_g.pth`), selective FT on the last 2 blocks (repo freeze loop)
- **IDPM** — Q-former + 32 query tokens (BERT-initialized, fully trainable) + CSRM context/gate layers
- **LLM** — Qwen3.5-2B-Base (Apache-2.0), **frozen**, LoRA r=16 α=32 on attention + MLP projections

Reuses the vendored `third_party/DeltaVLM` dataset/processor/Q-former/ViT code untouched; replaces the
Vicuna-7B decoder (non-commercial) with Qwen. The repo's `train.py`/configs are **not** used — they are
wired to Vicuna and `transformers==4.33.2`; the notebook re-implements the training loop against Qwen + PEFT.

## Run

1. Open `train_deltavlm_qwen.ipynb` on **Colab (A100 40 GB)** — recommended; **Kaggle is auto-detected**
   and switches to fp16 / batch 1 × accum 16 / grad-checkpoint (incl. Qwen) for you.
2. Edit only the CONFIG cell if needed — `hf_repo` defaults to `VMamidala/satquery-model-c-deltavlm-qwen`
   and the token comes from the `HF_TOKEN` Kaggle secret (Settings → Secrets).
3. `smoke_test=True` first → validates shapes (~1 min), then set `False` and run the full notebook.
4. **2xT4 sessions:** `use_ddp=True` is the default — the notebook writes a self-contained
   `train_ddp.py` and launches `torchrun --nproc_per_node=2` (effective batch 32). If torchrun fails on
   Kaggle, set `use_ddp=False` and the inline single-GPU cell runs instead (effective batch 16).

Data is downloaded automatically: annotations from `hlwu/changechat-105k` (CC-BY-4.0) and images from the
official LEVIR-CC distribution `lcybuaa/LEVIR-CC` (~2.7 GB zip; Google Drive/Baidu are the alternates listed
in the LEVIR-CC README). Do **not** install `third_party/DeltaVLM/requirements.txt` (pins transformers 4.33.2).

## Outputs (for IMP-036 import)

`<output_dir>/` (Kaggle: `/kaggle/working/satquery_models/model_c/`) holds one rolling checkpoint
`ckpt_latest/` — written and **overwritten after every epoch** (final state after the last epoch) — with:
`qwen_lora/` (PEFT adapter + tokenizer), `bi_ve_finetuned.pt`, `q_former_finetuned.pt`, `llm_proj.pt`
(including CSRM layers), plus `lineage.json` (base weights, data, hyperparams, gate scores, sha256 checksums).
Fine-tuned weights are stored **fp16** (halves artifact size; inference casts to its dtype anyway).
Only `ckpt_latest/` + `lineage.json` are pushed to HF Hub.

## Getting checkpoints onto your computer

Kaggle/Colab runtime disks are **ephemeral** — `/kaggle/working` is wiped when the session ends, so
artifacts must be pushed out before then. The notebook does this automatically:

1. Set your HF token as a **Kaggle secret** named `HF_TOKEN` (Settings → Secrets → Add new secret, name
   `HF_TOKEN`; on Colab: userdata).
2. Edit `CFG["hf_repo"]` in the CONFIG cell if you want a different repo name — default is
   `VMamidala/satquery-model-c-deltavlm-qwen` (created privately on first run).
3. During training, **after every epoch** `ckpt_latest/` is uploaded to that private HF repo automatically —
   same file paths each time, so they are overwritten on HF, never duplicated. The final "Auto-save via
   HF Hub" cell re-uploads the last state together with `lineage.json`. Then download on your computer:

       huggingface-cli download VMamidala/satquery-model-c-deltavlm-qwen --local-dir models/deltavlm-qwen-v2

No HF token? Commit the notebook instead: output is kept in the Kaggle output dataset and pulled with
`kaggle kernels output <owner>/<kernel> -p .`.

**Nothing is downloaded to your local machine by the notebook** — datasets and checkpoints live only in
the runtime (and the HF repo you pull from).

## Gates (IMP-035)

- Binary change classification Acc ≥ 90% on `changechat_105k_test_binary.json` (1,929 samples)
- Caption CIDEr ≥ baseline on `changechat_105k_test.json` (5 refs/pair) — **pin the DeltaVLM paper number
  before declaring the gate passed**; record, do not tune (DEC-021)
- CDVQA test1/test2 dry-run is **IMP-037**, measurement-only, intentionally not in this notebook

## Recorded decisions (see `docs/Decision_Log.md`)

1. **DEC-033** — "Qwen3.5-2B" exists on HF only as a VLM-family checkpoint (`Qwen3_5ForConditionalGeneration`);
   the notebook loads `Qwen/Qwen3.5-2B-Base` and uses its text-LM submodule as the frozen decoder.
   `config/model_profiles.yaml` pins `Qwen2.5-1.5B/3B-Instruct` (stale vs the doc) — kept as automatic fallback.
2. **DEC-034** — ChangeChat-105k has no official `val` split on HF; the IMP-035 gate is measured on the
   shipped test files (binary / caption), with no tuning. CDVQA dry-run remains IMP-037.

## License posture

DeltaVLM code Apache-2.0 · ChangeChat-105k annotations CC-BY-4.0 · Qwen3.5-2B Apache-2.0 ·
LEVIR-CC imagery under its own (free) license — cite Liu et al. 2022 (TGRS). No Vicuna weights used.
