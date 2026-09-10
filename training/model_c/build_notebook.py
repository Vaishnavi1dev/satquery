"""Builds training/model_c/train_deltavlm_qwen.ipynb - IMP-035 Model C fine-tune notebook.

Run:  python training/model_c/build_notebook.py
Output: training/model_c/train_deltavlm_qwen.ipynb
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "train_deltavlm_qwen.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    }


CELLS = []

# ----------------------------------------------------------------------------
CELLS.append(md(
"""# IMP-035 - Model C Fine-Tune: DeltaVLM + Qwen3.5-2B on ChangeChat-105k

**Task:** FT-C per `docs/Implementation_Plan.md` IMP-035 / `docs/Model_Selection.md` §3.3
**Architecture:** Bi-VE (EVA-ViT-g/14, selective FT last 2 blocks) + IDPM (CSRM + Q-former, FT) + Qwen3.5-2B (frozen, LoRA r=16, α=32)
**Data:** ChangeChat-105k train split (87,935 samples, CC-BY-4.0 annotations + LEVIR-CC bitemporal imagery)
**Target runtime:** Colab A100 40 GB (recommended) · T4 16 GB fallback (fp16, batch 1-2, grad checkpoint)

**License posture (all compliant):** DeltaVLM code Apache-2.0 · ChangeChat-105k annotations CC-BY-4.0 ·
Qwen3.5-2B Apache-2.0 · LEVIR-CC imagery under its own (free) license, not redistributed by us.
Do **not** train with the repo's original Vicuna-7B weights (non-commercial) - this notebook swaps in Qwen.

**Recorded implementation notes (see `docs/Decision_Log.md`):**
1. `Qwen/Qwen3.5-2B` on the Hub is a VLM-family checkpoint (`Qwen3_5ForConditionalGeneration`).
   We use `Qwen/Qwen3.5-2B-Base` and unwrap its **text LM submodule** as the frozen causal decoder -
   the faithful realization of "Qwen3.5-2B (Apache-2.0)" as DeltaVLM's LLM. `config/model_profiles.yaml`
   pins `Qwen2.5-1.5B/3B-Instruct` (stale vs the doc) - kept as the automatic fallback, not silently resolved.
2. ChangeChat-105k has **no official `val` split** on HF (train + 6 test files only). IMP-035's gate
   ("val: caption CIDEr ≥ baseline; binary Acc ≥90%") is measured on the shipped **test** files
   (`changechat_105k_test_binary.json`, `changechat_105k_test.json`) - recorded, no tuning.
   The CDVQA test1/test2 dry-run stays in IMP-037 (measurement only) and is NOT part of this notebook.

**Repo baseline:** code is reused from `third_party/DeltaVLM` (cloned from https://github.com/hanlinwu/DeltaVLM).
The stock `train.py`/configs are wired to Vicuna-7B + `transformers==4.33.2`; this notebook reuses the
**dataset / processor / Q-former / EVA-ViT** code as-is and replaces the LLM side with Qwen + PEFT.
"""))

# ----------------------------------------------------------------------------
CELLS.append(md(
"""## 0. Runtime & paths

Auto-detects Colab / Kaggle / local. All paths below flow from one CONFIG dict - edit that dict only.
Outputs (adapters, checkpoints, lineage) land in `OUTPUT_DIR`; artifacts are self-contained for IMP-036 import.
"""))

# ----------------------------------------------------------------------------
CELLS.append(code(
"""import os, sys, platform, json, math, time, hashlib, subprocess, shutil, random
import numpy as np
import torch

IN_COLAB = "google.colab" in sys.modules
IN_KAGGLE = os.path.isdir("/kaggle")

if IN_COLAB:
    BASE = "/content"
elif IN_KAGGLE:
    BASE = "/kaggle/working"
else:
    BASE = os.getcwd()

CFG = dict(
    # --- paths ---
    base_dir=BASE,
    deltavlm_dir=os.path.join(BASE, "third_party", "DeltaVLM"),   # cloned/found here
    dataset_dir=os.path.join(BASE, "dataset"),                    # annotations/ + images/ (repo layout)
    output_dir=os.path.join(BASE, "satquery_models", "model_c"),  # checkpoints + lineage
    # --- LLM decoder (Model_Selection §3.3: Qwen3.5-2B, Apache-2.0) ---
    llm_backbone="Qwen/Qwen3.5-2B-Base",      # text backbone of the Qwen3.5-2B family (unwrap language_model)
    llm_fallback="Qwen/Qwen2.5-3B-Instruct",  # config/model_profiles.yaml pin - used only if primary fails
    # --- LoRA on Qwen (r=16-32, alpha=32 per IMP-035) ---
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # --- Bi-VE selective FT: last 2 blocks (repo freeze loop keeps blocks.37/.38 trainable) ---
    vit_blocks_trainable={"blocks.37", "blocks.38"},
    grad_checkpoint=False,
    # --- IDPM / Q-former ---
    num_query_token=32,
    img_size=224,
    max_txt_len=128,
    max_output_txt_len=256,
    # --- optimization ---
    init_lr=1e-5,
    min_lr=0.0,
    warmup_steps=1000,
    weight_decay=0.05,
    grad_clip=1.0,
    batch_size=8,        # T4: 1
    accum_steps=4,       # T4: 16 (effective batch = batch_size * accum_steps)
    max_epochs=1,        # Kaggle default: 1 epoch (~8h, fits 12h limit); A100: 3
    dtype="bf16",        # A100: bf16 | T4: fp16 (auto-switched on Kaggle)
    num_workers=4,       # set 0 on Windows
    seed=42,
    # --- data / eval ---
    data_subset=None,    # int -> take first N train samples (quick runs); None = full 87,935
    eval_binary_max=1929,   # full test_binary split
    cider_subset=100,       # caption-test subset for the CIDEr/BLEU gate
    # --- optional: initialize Bi-VE/Q-former from the authors' checkpoint ---
    # OFF by default: checkpoint was trained alongside non-commercial Vicuna; verify license first (TBD-002).
    init_from_deltavlm_ckpt=False,
    deltavlm_ckpt="https://huggingface.co/hlwu/DeltaVLM/resolve/main/checkpoint_best.pth",
    # --- smoke test (1 fwd/bwd + 1 generation, then stop) ---
    smoke_test=False,
    # --- auto-save checkpoints to your computer (HF Hub private repo) ---
    # Kaggle/Colab runtime disks are ephemeral: after training, the final checkpoint + lineage are
    # pushed to a private HF repo, then pulled locally with `huggingface-cli download`.
    # Token source: Kaggle secret named "HF_TOKEN" (or Colab userdata / env HF_TOKEN).
    push_to_hub=True,
    hf_repo="VMamidala/satquery-model-c-deltavlm-qwen",  # your HF repo (private, auto-created)
    hf_token_secret="HF_TOKEN",
    save_steps=500,        # overwrite ckpt_latest & push to HF every 500 steps (None = epoch only)
    resume_from_hf=True,   # pull ckpt_latest from CFG['hf_repo'] before training
    resume_checkpoint_dir=None, # local dir to resume weights from (or set by resume_from_hf)
    start_step=1300,       # step number to resume from (skips seen batches and advances scheduler)
    # --- multi-GPU ---
    use_ddp=True,      # use ALL visible GPUs (Kaggle '2x T4 GPU') via torchrun; False = single GPU
    ddp_done=False,    # set True by the DDP cell after a successful torchrun run
)

# Kaggle runtime = T4 16 GB: fp16, batch 1 x accum 16, grad checkpoint (incl. Qwen LLM).
# max_epochs=1 takes ~7.5-8.5h on ChangeChat-105k, fitting reliably within Kaggle's 12h session limit.
# On a 2xT4 session DDP doubles the effective batch to 32; on 1 GPU it stays 16.
IS_T4 = torch.cuda.is_available() and any("T4" in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
if IN_KAGGLE or IS_T4:
    CFG.update(batch_size=1, accum_steps=16, dtype="fp16", grad_checkpoint=True, num_workers=2, max_epochs=1)
    print("T4 / Kaggle detected - T4-safe defaults applied: fp16, batch 1, accum 16, grad checkpoint, max_epochs 1 (fits 12h limit)")

os.makedirs(CFG["output_dir"], exist_ok=True)
random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("runtime:", "colab" if IN_COLAB else ("kaggle" if IN_KAGGLE else "local"))
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
gb = shutil.disk_usage(BASE)
print(f"free disk: {gb.free/1e9:.1f} GB (need ~15 GB for weights+data)")
"""))

# ----------------------------------------------------------------------------
CELLS.append(md(
"""## 1. Install dependencies

**Do NOT** `pip install -r third_party/DeltaVLM/requirements.txt` - it pins `transformers==4.33.2`,
which breaks Qwen loading. Install the modern stack below (Qwen3.5 needs a current transformers).
If the vendored `model/Qformer.py` import fails on a bleeding-edge transformers release, pin
`transformers==4.46.3` and re-run this cell.

> **Pillow:** the cell below verifies the real `PIL.ImageText` import path (Kaggle's Pillow can be
> internally mismatched and break torchvision/timm). If it has to pin Pillow 11.3.0, **restart the
> kernel** (Runtime → Restart) and re-run from this cell before continuing.
"""))

# ----------------------------------------------------------------------------
CELLS.append(code(
"""get_ipython().system('pip install -q -U transformers peft accelerate bitsandbytes sentencepiece timm omegaconf easydict pycocotools nltk pandas scipy webdataset iopath opencv-python-headless tqdm requests')

# Pillow sanity: some Kaggle Pillow builds are internally inconsistent (PIL.ImageDraw -> ImageText
# imports _Ink from PIL._typing and fails), which breaks the torchvision -> timm import chain.
# Test the REAL chain and pin a known-good Pillow if broken. If a pin happens here, RESTART THE
# KERNEL (Runtime -> Restart) and re-run from this cell before continuing.
import PIL

def _pillow_broken():
    try:
        import PIL.ImageDraw      # on Pillow 12 this also pulls in ImageText
        return False
    except ModuleNotFoundError:
        return False              # Pillow < 12: ImageText doesn't exist - harmless
    except Exception:
        return True               # inconsistent build (e.g. `_Ink` ImportError)

if _pillow_broken():
    print("Pillow build broken on %s - pinning 11.3.0" % PIL.__version__)
    get_ipython().system('pip install -q --force-reinstall "pillow==11.3.0"')
    print("Pillow pinned. RESTART THE KERNEL (Runtime -> Restart), then re-run from this cell.")
else:
    print("Pillow OK:", PIL.__version__)

# PEFT probes torchao for LoRA dispatch; Kaggle ships an old torchao (0.10.x) that modern peft
# rejects (wants >=0.16). We don't use torchao quantization - remove it so peft uses the standard path.
import importlib.util as _ilu
if _ilu.find_spec("torchao"):
    get_ipython().system('pip uninstall -y -q torchao')
    print("torchao uninstalled (incompatible with peft; not needed for LoRA training)")

import importlib.metadata as im
for p in ["torch", "transformers", "peft", "accelerate", "timm", "pillow", "torchvision"]:
    try: print(p, im.version(p))
    except Exception as e: print(p, "MISSING", e)
"""))

# ----------------------------------------------------------------------------
CELLS.append(code(
"""# Fetch DeltaVLM code (public, Apache-2.0) if not already present.
DELTA = CFG["deltavlm_dir"]
if not os.path.isdir(os.path.join(DELTA, "model")):
    os.makedirs(os.path.dirname(DELTA), exist_ok=True)
    get_ipython().system(f'git clone --depth 1 https://github.com/hanlinwu/DeltaVLM.git "{DELTA}"')

sys.path.insert(0, DELTA)
sys.path.insert(0, os.path.dirname(DELTA))   # so `import utils` (DeltaVLM root) resolves

# The vendored code loads bert-base-uncased from "../bert-base-uncased" - a path RELATIVE TO CWD.
# chdir into the repo so that resolves to <parent>/bert-base-uncased (downloaded in the next cell).
# All notebook paths are absolute, so this is safe.
os.chdir(DELTA)
print("cwd ->", os.getcwd())

try:
    sha = subprocess.run(["git", "-C", DELTA, "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
except Exception:
    sha = "unknown"
print("DeltaVLM code at:", DELTA, "| commit:", sha)
CFG["deltavlm_commit"] = sha
"""))

# ----------------------------------------------------------------------------
CELLS.append(code(
"""# timm compatibility shim - the vendored code was written against timm 0.4.12
# (`import timm.models.hub`, `from timm.models.layers import drop_path, to_2tuple, trunc_normal_`).
# This makes those imports resolve on any modern timm, and implements the hub downloader directly.
import types

def _make_hub_module():
    try:
        import timm.models.hub as hub            # noqa: F401  (modern timm keeps it)
        return hub
    except Exception:
        pass
    import timm, os as _os
    from urllib.parse import urlparse
    import requests

    def _get_cache_dir():
        d = _os.environ.get("TIMM_CACHE_DIR") or _os.path.join(_os.path.expanduser("~"), ".cache", "timm")
        _os.makedirs(d, exist_ok=True)
        return d

    def _download_cached_file(url, check_hash=True, progress=False):
        dst = _os.path.join(_get_cache_dir(), _os.path.basename(urlparse(url).path))
        if not _os.path.exists(dst):
            print("downloading", url)
            r = requests.get(url, stream=True, timeout=600)
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        return dst

    m = types.ModuleType("timm.models.hub")
    m.get_cache_dir = _get_cache_dir
    m.download_cached_file = _download_cached_file
    return m

_timm_hub = _make_hub_module()
sys.modules["timm.models.hub"] = _timm_hub   # eva_vit.py does `import timm.models.hub as timm_hub`

if "timm.models.layers" not in sys.modules:
    try:
        from timm.models.layers import drop_path, to_2tuple, trunc_normal_   # noqa: F401
    except Exception:
        from timm.layers import drop_path, to_2tuple, trunc_normal_          # noqa: F401
        _ml = types.ModuleType("timm.models.layers")
        _ml.drop_path, _ml.to_2tuple, _ml.trunc_normal_ = drop_path, to_2tuple, trunc_normal_
        sys.modules["timm.models.layers"] = _ml
print("timm shim OK")
"""))

# ----------------------------------------------------------------------------
CELLS.append(code(
"""# Vendored LAVIS Qformer compat shim - model/Qformer.py was written against transformers 4.x, which
# exported apply_chunking_to_forward etc. from transformers.modeling_utils. Newer transformers (5.x,
# required for Qwen3.5) removed some of them. Inject compatible fallbacks before importing the model.
import torch
import torch.nn as nn
import transformers.modeling_utils as _mu


def _inject(name, fn):
    if not hasattr(_mu, name):
        setattr(_mu, name, fn)
        print(f"[compat] injected transformers.modeling_utils.{name}")


def _apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
    # chunking is a memory optimization; BERT chunk_size_feed_forward defaults to 0 (disabled),
    # so calling forward directly is equivalent for our Q-former
    return forward_fn(*input_tensors)


def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head -= sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index


def _prune_linear_layer(layer, index, dim=0):
    index = index.to(layer.weight.device)
    if dim == 0:
        new = nn.Linear(layer.in_features, len(index), bias=layer.bias is not None)
        new.weight.data = layer.weight.data[index, :].clone()
        if layer.bias is not None:
            new.bias.data = layer.bias.data[index].clone()
    elif dim == 1:
        new = nn.Linear(len(index), layer.out_features, bias=layer.bias is not None)
        new.weight.data = layer.weight.data[:, index].clone()
        if layer.bias is not None:
            new.bias.data = layer.bias.data.clone()
    new.weight.requires_grad_(False)
    if layer.bias is not None:
        new.bias.requires_grad_(False)
    return new


_inject("apply_chunking_to_forward", _apply_chunking_to_forward)
_inject("find_pruneable_heads_and_indices", _find_pruneable_heads_and_indices)
_inject("prune_linear_layer", _prune_linear_layer)
print("Qformer compat shim done")

# --- Qformer BERT-init bypass ---
# transformers v5's PreTrainedModel.from_pretrained internals (mark_tied_weights_as_initialized /
# all_tied_weights_keys) break the vendored LAVIS BertLMHeadModel. Build the Q-former from config and
# copy bert-base-uncased weights manually; cross-attention layers stay random and are trained (BLIP-2
# stage-1 style). Identical result, no v5 loading machinery involved.
import os as _os
from model import blip2 as _blip2
from model.Qformer import (
    BertConfig as _BertConfig,
    BertPreTrainedModel as _BertPreTrainedModel,
    BertLMHeadModel as _BertLMHeadModel,
)

# cheap insurance for v5's tied-weights machinery on these pre-v5 classes
# (v5 PreTrainedModel.init_weights -> tie_weights reads `all_tied_weights_keys` as a dict,
# and tie_weights itself must not try to tie anything since we load weights manually)
_BertPreTrainedModel.all_tied_weights_keys = {}
_BertPreTrainedModel._tied_weights_keys = {}
_BertPreTrainedModel.tie_weights = lambda self, *args, **kwargs: None


def _patched_init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
    encoder_config = _BertConfig.from_pretrained("../bert-base-uncased")
    encoder_config.encoder_width = vision_width
    encoder_config.add_cross_attention = True
    encoder_config.cross_attention_freq = cross_attention_freq
    encoder_config.query_length = num_query_token
    Qformer = _BertLMHeadModel(encoder_config)
    bert_sd = torch.load(_os.path.join("..", "bert-base-uncased", "pytorch_model.bin"), map_location="cpu")
    missing, unexpected = Qformer.load_state_dict(bert_sd, strict=False)
    print(f"[compat] Qformer built from config + BERT weights: {len(missing)} missing "
          f"(cross-attn etc., trained later), {len(unexpected)} unexpected")
    query_tokens = torch.nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
    query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    return Qformer, query_tokens


_blip2.Blip2Base.init_Qformer = classmethod(_patched_init_Qformer)
print("Qformer init patched - manual BERT weight load")

# --- v4-era PreTrainedModel helper methods used by the vendored BertModel.forward ---
# transformers v5 removed get_head_mask / invert_attention_mask from PreTrainedModel; re-add them
# on the vendored base class (implementations copied from transformers v4 modeling_utils).


def _add_method(cls, name, fn):
    if not hasattr(cls, name):
        setattr(cls, name, fn)
        print(f"[compat] added {cls.__name__}.{name}")


def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
    if head_mask is not None:
        head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked:
            head_mask = head_mask.unsqueeze(-1)
    else:
        head_mask = [None] * num_hidden_layers
    return head_mask


def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    assert head_mask.dim() == 5, f"head_mask.dim() != 5, instead {head_mask.dim()}"
    head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
    return head_mask


def _invert_attention_mask(self, encoder_attention_mask):
    if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    elif encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    else:
        raise ValueError(f"wrong shape for encoder_attention_mask: {encoder_attention_mask.shape}")
    dtype = next(self.parameters()).dtype
    encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=dtype)
    if getattr(self.config, "is_decoder", False):
        device = encoder_attention_mask.device
        batch_size, seq_length = encoder_attention_mask.shape
        seq_ids = torch.arange(seq_length, device=device)
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        causal_mask = causal_mask.to(dtype)
        if causal_mask.shape[1] < encoder_extended_attention_mask.shape[1]:
            prefix = encoder_extended_attention_mask.shape[1] - causal_mask.shape[1]
            causal_mask = torch.cat(
                [torch.ones((batch_size, seq_length, prefix), device=device, dtype=dtype), causal_mask],
                dim=-1)
        encoder_extended_attention_mask = encoder_extended_attention_mask * causal_mask[:, None, :, :]
    encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(dtype).min
    return encoder_extended_attention_mask


_add_method(_BertPreTrainedModel, "get_head_mask", _get_head_mask)
_add_method(_BertPreTrainedModel, "_convert_head_mask_to_5d", _convert_head_mask_to_5d)
_add_method(_BertPreTrainedModel, "invert_attention_mask", _invert_attention_mask)
"""))

# ----------------------------------------------------------------------------
CELLS.append(code(
"""# The vendored Blip2Base loads bert-base-uncased + the Q-former from the LOCAL path
# "../bert-base-uncased" relative to the DeltaVLM dir - download it there once.
from huggingface_hub import snapshot_download

bert_dir = os.path.join(os.path.dirname(CFG["deltavlm_dir"]), "bert-base-uncased")
if not os.path.isdir(bert_dir):
    print("downloading bert-base-uncased ->", bert_dir)
    snapshot_download("bert-base-uncased", local_dir=bert_dir)
print("bert-base-uncased:", len(os.listdir(bert_dir)), "files")
"""))

# ----------------------------------------------------------------------------
CELLS.append(md(
"""## 2. Data - ChangeChat-105k (annotations) + LEVIR-CC (images)

- Annotations: `hlwu/changechat-105k` (CC-BY-4.0, ~83 MB) - all JSONs, incl. the 87,935-sample train file.
- Images: `lcybuaa/LEVIR-CC` (the **official authors'** distribution - also linked from the LEVIR-CC
  README; Google Drive/Baidu are the alternates). Single `Levir-CC-dataset.zip` (~2.7 GB) containing
  `images/{train,val,test}/{A,B}/` bitemporal tiles. Layout is auto-detected after unzip.
"""))

# ----------------------------------------------------------------------------
CELLS.append(code(
"""from huggingface_hub import snapshot_download

DS = CFG["dataset_dir"]
ann_dir = os.path.join(DS, "annotations")
img_root = os.path.join(DS, "images")
os.makedirs(ann_dir, exist_ok=True)

# 1) annotations
if not os.listdir(ann_dir):
    print("downloading changechat-105k annotations...")
    src = snapshot_download("hlwu/changechat-105k", repo_type="dataset")
    for f in os.listdir(src):
        if f.endswith(".json"):
            shutil.copy(os.path.join(src, f), os.path.join(ann_dir, f))
print("annotations:", sorted(os.listdir(ann_dir)))

# 2) images (LEVIR-CC zip from the official HF mirror)
zip_path = os.path.join(DS, "Levir-CC-dataset.zip")
if not os.path.exists(zip_path) and not os.path.isdir(os.path.join(img_root, "train")):
    print("downloading LEVIR-CC (2.7 GB)...")
    snapshot_download("lcybuaa/LEVIR-CC", repo_type="dataset",
                      allow_patterns=["*.zip"], local_dir=DS)
    zips = [f for f in os.listdir(DS) if f.endswith(".zip")]
    assert zips, "LEVIR-CC zip not found"
    zip_path = os.path.join(DS, zips[0])

if os.path.exists(zip_path) and not os.path.isdir(os.path.join(img_root, "train")):
    print("unzipping", zip_path)
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(os.path.join(DS, "levir_extract"))

# 3) locate the images/ root wherever the zip put it
def find_images_root(base):
    for root, dirs, files in os.walk(base):
        if os.path.basename(root) == "images" and os.path.isdir(os.path.join(root, "train", "A")):
            return root
    return None

found = find_images_root(DS)
if found and not os.path.isdir(img_root):
    shutil.move(found, img_root)

# Unconditional cleanup of intermediate extract directory, zip files, and HF download cache
shutil.rmtree(os.path.join(DS, "levir_extract"), ignore_errors=True)
for f in os.listdir(DS):
    if f.endswith(".zip"):
        try: os.remove(os.path.join(DS, f))
        except Exception: pass
shutil.rmtree(os.path.expanduser("~/.cache/huggingface/hub/datasets--lcybuaa--LEVIR-CC"), ignore_errors=True)

assert os.path.isdir(os.path.join(img_root, "train", "A")), "LEVIR-CC images/train/A not found"
nA = len(os.listdir(os.path.join(img_root, "train", "A")))
nB = len(os.listdir(os.path.join(img_root, "train", "B")))
print(f"images root: {img_root} | train A/B pairs: {nA}/{nB} (expect 7590)")
CFG["vis_root"] = img_root
CFG["ann_dir"] = ann_dir
"""))

# ----------------------------------------------------------------------------
CELLS.append(code(
"""# Data sanity: counts + one full record.
import json
train_ann = json.load(open(os.path.join(CFG["ann_dir"], "changechat_105k_train.json")))
print("train records:", len(train_ann), "(expect 87935)")
r = train_ann[0]
print("sample:", r["id"], r["image"], "| changeflag:", r["changeflag"])
print("Q:", r["conversations"][0]["value"][:100])
print("A:", r["conversations"][1]["value"][:100])
from PIL import Image
for p in r["image"]:
    im = Image.open(os.path.join(CFG["vis_root"], p)).convert("RGB")
    print(p, im.size)
CFG["train_ann"] = train_ann
"""))

# ----------------------------------------------------------------------------
CELLS.append(md(
"""## 3. Dataset + processors (reusing the repo's classes)

`CaptionDataset` (from DeltaVLM `dataset.py`) expands each record's `conversations` into
(question, answer) training instances - exactly the repo's training pipeline. Processors:
`Blip2ImageTrainProcessor` (224 px) for images and `BlipCaptionProcessor` for text.
"""))

# ----------------------------------------------------------------------------
CELLS.append(code(
"""from processor import Blip2ImageTrainProcessor, BlipImageEvalProcessor, BlipCaptionProcessor
from dataset import CaptionDataset

train_vis = Blip2ImageTrainProcessor(image_size=CFG["img_size"])
eval_vis = BlipImageEvalProcessor(image_size=CFG["img_size"])
txt_proc = BlipCaptionProcessor()

train_path = os.path.join(CFG["ann_dir"], "changechat_105k_train.json")
train_ds = CaptionDataset(vis_processor=train_vis, text_processor=txt_proc,
                          vis_root=CFG["vis_root"], ann_paths=[train_path])
if CFG["data_subset"]:
    train_ds.annotation = train_ds.annotation[: CFG["data_subset"]]
print("train samples:", len(train_ds))

s = train_ds[0]
print("image_A:", tuple(s["image_A"].shape), "image_B:", tuple(s["image_B"].shape))
print("text_input:", s["text_input"][:80])
print("text_output:", s["text_output"][:80])
"""))

# ----------------------------------------------------------------------------
CELLS.append(md(
"""## 4. Model - Qwen-adapted DeltaVLM

Reuses from the vendored code, unmodified:
- `Blip2Base.init_vision_encoder` → EVA-ViT-g/14 (pretrained `eva_vit_g.pth`, auto-downloaded via timm shim)
- `Blip2Base.init_Qformer` → LAVIS BERT-based Q-former + 32 learnable query tokens
- freeze loop keeps the **last 2 ViT blocks** trainable (selective FT per IMP-035)

Replaces the Vicuna-7B (non-commercial) decoder with **Qwen3.5-2B-Base, frozen + LoRA**:
- `Qwen3.5-2B` on HF is a VLM-family checkpoint → we load it and unwrap its text-LM submodule
  (`language_model`); plain causal-LM checkpoints (e.g. the `Qwen2.5-3B-Instruct` profile pin) load directly.
- Forward/generate flows are copied verbatim from `model/blip2_vicua.py`: train forward = concatenated
  bi-temporal features through the Q-former (text-conditioned), projected by `llm_proj` and prepended to
  the LLM's token embeddings; `generate` additionally applies the CSRM difference-perception layers
  (context/gate) before the Q-former - matching the authors' code exactly.
"""))

# ----------------------------------------------------------------------------
CELLS.append(code(
"""from model.blip2 import Blip2Base, disabled_train
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from transformers import AutoModelForImageTextToText
except Exception:  # older transformers - will fall back to AutoModel
    AutoModelForImageTextToText = None
from peft import LoraConfig, get_peft_model
import torch.nn as nn


def load_qwen_decoder(backbone, fallback, dtype):
    \"\"\"Load the frozen text decoder. Primary: Qwen3.5-2B-Base (unwrap text LM).
    Fallback: plain causal LM (model_profiles.yaml pin).\"\"\"
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_lm, last_err = None, None
    for repo in (backbone, fallback):
        try:
            try:
                model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
                text_lm = model
            except Exception as e1:
                if AutoModelForImageTextToText is not None:
                    model = AutoModelForImageTextToText.from_pretrained(repo, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
                else:
                    from transformers import AutoModel
                    model = AutoModel.from_pretrained(repo, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
                text_lm = getattr(model, "language_model", None) or getattr(model, "model", None)
                if text_lm is None or not hasattr(text_lm, "get_input_embeddings"):
                    raise RuntimeError(f"no usable text-LM submodule in {repo}")
            print(f"[decoder] loaded {repo} -> text LM with hidden_size={text_lm.config.hidden_size}")
            break
        except Exception as e:
            last_err = e
            print(f"[decoder] failed on {repo}: {type(e).__name__}: {e}")
            continue
    if text_lm is None:
        raise RuntimeError(f"could not load any decoder. last error: {last_err}")
    for p in text_lm.parameters():
        p.requires_grad = False
    return tokenizer, text_lm


def find_lora_targets(model, candidates):
    found = {n.split(".")[-1] for n, _ in model.named_modules() if n.split(".")[-1] in candidates}
    return sorted(found) or ["q_proj", "v_proj"]


def apply_lora(llm, cfg):
    targets = find_lora_targets(llm, cfg["lora_targets"])
    lora_cfg = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                          lora_dropout=cfg["lora_dropout"], target_modules=targets,
                          bias="none", task_type="CAUSAL_LM")
    return get_peft_model(llm, lora_cfg)


class DeltaVLMQwen(Blip2Base):
    \"\"\"DeltaVLM (Bi-VE + CSRM + Q-former) with a frozen Qwen3.5-2B LoRA decoder.\"\"\"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dt = torch.bfloat16 if cfg["dtype"] == "bf16" else torch.float16
        self.amp_dtype = dt

        # 1) Q-former-side BERT tokenizer ([DEC] added, as in the repo)
        self.tokenizer = self.init_tokenizer(truncation_side="left")

        # 2) Bi-temporal vision encoder - EVA-ViT-g/14, selective FT on the last 2 blocks.
        # create_eva_vit_g loads fp16; the repo's `.float()` then undoes that and costs ~2.4 GB,
        # which OOMs a 16 GB T4. Keep the ViT in the training dtype instead (fp16 on T4 / bf16 on
        # A100) - the forward runs under autocast, so fp32 params are not needed.
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            "eva_clip_g", cfg["img_size"], 0.0, False, "fp16")
        # NOTE: use_grad_checkpoint (last arg) is hardcoded False - the vendored EVA-ViT calls
        # torch.utils.checkpoint with legacy reentrant semantics, which raises CheckpointError on
        # backward with modern torch (saved/recomputed tensor counts diverge). The ViT's activations
        # are only ~0.3-0.5 GB, so materializing them costs far less than fighting that bug.
        # Qwen keeps its own (modern, use_reentrant=False) gradient checkpointing via cfg below.
        vit_dtype = torch.bfloat16 if cfg["dtype"] == "bf16" else torch.float16
        self.visual_encoder = self.visual_encoder.to(dtype=vit_dtype)
        self.ln_vision = self.ln_vision.float()  # tiny (1408 params); vendored LayerNorm casts anyway
        # T4/fp16: cuDNN v9 can fail to allocate a workspace for the fp16 patch-embed conv
        # (CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED) even with memory free. EVA-ViT has
        # exactly ONE conv (the patch embed, 3->1408) - disabling cuDNN here costs ~nothing and
        # routes that conv through torch's own kernel instead.
        torch.backends.cudnn.enabled = False
        for name, param in self.visual_encoder.named_parameters():
            if any(b in name for b in cfg["vit_blocks_trainable"]):
                continue
            param.requires_grad = False
        self.visual_encoder.eval()
        self.visual_encoder.train = disabled_train
        n_train = sum(1 for n, p in self.visual_encoder.named_parameters() if p.requires_grad)
        print(f"[bi-ve] {n_train} params trainable (last-2-block selective FT)")

        # 3) IDPM: Q-former + query tokens (fully trainable)
        self.Qformer, self.query_tokens = self.init_Qformer(
            cfg["num_query_token"], self.visual_encoder.num_features)
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        # 4) CSRM difference-perception layers (used in generate(), as in the repo)
        nf = self.visual_encoder.num_features
        self.context1 = nn.Linear(nf, nf, bias=False)
        self.context2 = nn.Linear(nf, nf)
        self.gate1 = nn.Linear(nf, nf, bias=False)
        self.gate2 = nn.Linear(nf, nf)
        self.dropout = nn.Dropout(0.5)
        self.context3 = nn.Linear(3 * nf, nf)
        # CSRM difference layers are only used in generate(), not in training forward().
        # Freeze them during training so DDP can use find_unused_parameters=False
        # without conflicting with gradient checkpointing on Qwen.
        for m in (self.context1, self.context2, self.gate1, self.gate2, self.context3):
            for p in m.parameters():
                p.requires_grad = False

        # 5) Qwen decoder: frozen base + LoRA adapters
        self.llm_tokenizer, llm_model = load_qwen_decoder(
            cfg["llm_backbone"], cfg["llm_fallback"], cfg["dtype"])
        llm_hidden = llm_model.config.hidden_size
        self.llm_model = apply_lora(llm_model, cfg)
        self.llm_model.print_trainable_parameters()
        if cfg["grad_checkpoint"]:
            try:
                self.llm_model.gradient_checkpointing_enable()
                self.llm_model.enable_input_require_grads()
                self.llm_model.config.use_cache = False
                print("[llm] gradient checkpointing enabled (use_cache=False)")
            except Exception as e:
                print("[llm] grad-checkpoint enable failed (non-fatal):", e)
        self.llm_proj = nn.Linear(self.Qformer.config.hidden_size, llm_hidden)

        self.max_txt_len = cfg["max_txt_len"]
        self.max_output_txt_len = cfg["max_output_txt_len"]
        self.prompt = ""

        # Required for T4 / fp16 mixed precision with GradScaler:
        # Trainable parameters must be fp32 master weights so gradients accumulate in fp32.
        # Otherwise scaler.unscale_() raises: ValueError: Attempting to unscale FP16 gradients.
        for p in self.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()

    # -- copied from model/blip2_vicua.py, adapted for Qwen tokenizer -------
    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len, llm_tokens = [], {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            # Qwen tokenizer does not prepend a BOS token (unlike Vicuna/LLaMA's output_ids[i][1:])
            # Retain all output tokens so single-word answers like 'yes'/'no' are not sliced away.
            llm_tokens["input_ids"].append(torch.cat([
                input_ids[i][:this_input_ones], output_ids[i], input_ids[i][this_input_ones:]]))
            llm_tokens["attention_mask"].append(torch.cat([
                input_atts[i][:this_input_ones], output_atts[i], input_atts[i][this_input_ones:]]))
        llm_tokens["input_ids"] = torch.stack(llm_tokens["input_ids"])
        llm_tokens["attention_mask"] = torch.stack(llm_tokens["attention_mask"])
        return llm_tokens, input_part_targets_len

    def _autocast(self):
        return torch.autocast("cuda", dtype=self.amp_dtype) if torch.cuda.is_available() \
            else __import__("contextlib").nullcontext()

    # -- training forward (repo path: concatenated bi-temporal features) -----
    def forward(self, samples):
        imageA, imageB = samples["image_A"], samples["image_B"]
        with self._autocast():
            input_bef = self.ln_vision(self.visual_encoder(imageA))
            input_aft = self.ln_vision(self.visual_encoder(imageB))
            image_embeds = torch.cat((input_bef, input_aft), dim=1)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=imageA.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        text_Qformer = self.tokenizer(samples["text_input"], padding="longest", truncation=True,
                                      max_length=self.max_txt_len, return_tensors="pt").to(imageA.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=imageA.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)
        query_output = self.Qformer.bert(
            text_Qformer.input_ids, attention_mask=Qformer_atts, query_embeds=query_tokens,
            encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts, return_dict=True)

        inputs_llm = self.llm_proj(query_output.last_hidden_state[:, : query_tokens.size(1), :])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long, device=imageA.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = "left"
        text_input_tokens = self.llm_tokenizer(
            samples["text_input"], return_tensors="pt", padding="longest", truncation=True,
            max_length=self.max_txt_len).to(imageA.device)
        self.llm_tokenizer.truncation_side = "right"
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples["text_output"]],
            return_tensors="pt", padding="longest", truncation=True,
            max_length=self.max_output_txt_len).to(imageA.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids, text_input_tokens.attention_mask,
            text_output_tokens.input_ids, text_output_tokens.attention_mask)

        # Build target labels: mask prompt tokens and padding positions with -100
        targets = llm_tokens["input_ids"].clone()
        targets[llm_tokens["attention_mask"] == 0] = -100
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100
        empty_targets = torch.ones(atts_llm.size(), dtype=torch.long, device=imageA.device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)

        # Forward through decoder under autocast WITHOUT computing loss in fp16
        with self._autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens["input_ids"])
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens["attention_mask"]], dim=1)
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=None
            )

        # Compute Cross-Entropy Loss in FP32 outside autocast (prevents 65,504 overflow on T4)
        with torch.autocast("cuda", enabled=False) if torch.cuda.is_available() else __import__("contextlib").nullcontext():
            logits = outputs.logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            # Numerical guard: if batch has no valid supervised tokens, avoid 0/0=nan
            if torch.isnan(loss) or (shift_labels != -100).sum() == 0:
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        return {"loss": loss}

    # -- generation (repo path incl. CSRM difference perception) -------------
    @torch.no_grad()
    def generate(self, samples, use_nucleus_sampling=False, num_beams=1, max_length=None,
                 min_length=1, top_p=0.9, repetition_penalty=1.5, length_penalty=1,
                 num_captions=1, temperature=1):
        self.llm_tokenizer.padding_side = "left"
        imageA, imageB = samples["image_A"], samples["image_B"]
        bs = imageA.size(0)
        prompt = samples.get("prompt") or [self.prompt] * bs
        if isinstance(prompt, str):
            prompt = [prompt] * bs
        query_tokens = self.query_tokens.expand(bs, -1, -1)
        text_Qformer = self.tokenizer(prompt, padding="longest", truncation=True,
                                      max_length=self.max_txt_len, return_tensors="pt").to(imageA.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=imageA.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        with self._autocast():
            input_bef = self.ln_vision(self.visual_encoder(imageA))
            input_aft = self.ln_vision(self.visual_encoder(imageB))
            input_diff = input_aft - input_bef

            input_bef_context = torch.tanh(self.context1(input_diff) + self.context2(input_bef))
            input_bef_context = self.dropout(input_bef_context)
            input_bef_gate = torch.sigmoid(self.gate1(input_diff) + self.gate2(input_bef))
            input_bef_gate = self.dropout(input_bef_gate)
            input_befs = input_bef_gate * input_bef_context

            input_aft_context = torch.tanh(self.context1(input_diff) + self.context2(input_aft))
            input_aft_context = self.dropout(input_aft_context)
            input_aft_gate = torch.sigmoid(self.gate1(input_diff) + self.gate2(input_aft))
            input_aft_gate = self.dropout(input_aft_gate)
            input_afts = input_aft_gate * input_aft_context

            input_bef = input_bef.permute(0, 2, 1)
            input_aft = input_aft.permute(0, 2, 1)
            input_befs = input_befs.permute(0, 2, 1)
            input_afts = input_afts.permute(0, 2, 1)
            input_diff = input_diff.permute(0, 2, 1)

            input_before = torch.cat([input_bef, input_diff, input_befs], 1).permute(0, 2, 1)
            input_after = torch.cat([input_aft, input_diff, input_afts], 1).permute(0, 2, 1)
            image_embedsA = self.context3(input_before)
            image_embedsB = self.context3(input_after)
            image_embeds = torch.cat((image_embedsA, image_embedsB), dim=1)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=imageA.device)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids, attention_mask=Qformer_atts, query_embeds=query_tokens,
                encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts, return_dict=True)

            inputs_llm = self.llm_proj(query_output.last_hidden_state[:, : query_tokens.size(1), :])
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long, device=imageA.device)

            llm_tokens = self.llm_tokenizer(prompt, padding="longest", return_tensors="pt").to(imageA.device)
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)
            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                do_sample=use_nucleus_sampling, top_p=top_p, temperature=temperature,
                num_beams=num_beams, max_length=max_length or self.max_output_txt_len,
                min_length=min_length, repetition_penalty=repetition_penalty,
                length_penalty=length_penalty, num_return_sequences=num_captions)

        out = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [t.strip() for t in out]

    def load_checkpoint(self, ckpt_dir):
        \"\"\"Load fine-tuned weights from a saved checkpoint folder.\"\"\"
        import os, torch
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        lora_dir = os.path.join(ckpt_dir, "qwen_lora")
        sf_path = os.path.join(lora_dir, "adapter_model.safetensors")
        bin_path = os.path.join(lora_dir, "adapter_model.bin")
        if os.path.exists(sf_path):
            set_peft_model_state_dict(self.llm_model, load_file(sf_path))
            print(f"[load_ckpt] loaded LoRA adapter from {sf_path}")
        elif os.path.exists(bin_path):
            set_peft_model_state_dict(self.llm_model, torch.load(bin_path, map_location="cpu"))
            print(f"[load_ckpt] loaded LoRA adapter from {bin_path}")

        ve_pt = os.path.join(ckpt_dir, "bi_ve_finetuned.pt")
        if os.path.exists(ve_pt):
            s1 = torch.load(ve_pt, map_location="cpu")
            self.visual_encoder.load_state_dict(s1["visual_encoder"], strict=False)
            self.ln_vision.load_state_dict(s1["ln_vision"])
            print(f"[load_ckpt] loaded bi_ve from {ve_pt}")

        qf_pt = os.path.join(ckpt_dir, "q_former_finetuned.pt")
        if os.path.exists(qf_pt):
            s2 = torch.load(qf_pt, map_location="cpu")
            self.Qformer.load_state_dict(s2["Qformer"])
            self.query_tokens.data.copy_(s2["query_tokens"])
            print(f"[load_ckpt] loaded Qformer & query_tokens from {qf_pt}")

        proj_pt = os.path.join(ckpt_dir, "llm_proj.pt")
        if os.path.exists(proj_pt):
            s3 = torch.load(proj_pt, map_location="cpu")
            self.llm_proj.load_state_dict(s3["llm_proj"])
            for n in ("context1", "context2", "gate1", "gate2", "context3"):
                getattr(self, n).load_state_dict(s3[n])
            print(f"[load_ckpt] loaded llm_proj & CSRM layers from {proj_pt}")
"""))

# ----------------------------------------------------------------------------
CELLS.append(code(
"""model = DeltaVLMQwen(CFG)
if CFG.get("resume_checkpoint_dir") and os.path.exists(CFG["resume_checkpoint_dir"]):
    model.load_checkpoint(CFG["resume_checkpoint_dir"])
model = model.cuda() if torch.cuda.is_available() else model

# trainable-parameter census per component (for the lineage record)
groups = {}
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if "visual_encoder" in n: g = "bi_ve_last2"
    elif "Qformer" in n or "query_tokens" in n: g = "qformer"
    elif "llm_proj" in n: g = "llm_proj"
    elif "context" in n or "gate" in n: g = "csrm"
    elif "lora_" in n: g = "qwen_lora"
    else: g = "other"
    groups[g] = groups.get(g, 0) + p.numel()
for g, c in sorted(groups.items()):
    print(f"{g:12s} {c/1e6:8.2f} M params")
print("trainable total: %.2f M" % (sum(groups.values()) / 1e6))
CFG["trainable_groups"] = groups
"""))

# ----------------------------------------------------------------------------
CELLS.append(code(
"""# OPTIONAL (default OFF): initialize Bi-VE + Q-former + CSRM from the authors' DeltaVLM checkpoint.
# Caveat: that checkpoint was trained alongside the non-commercial Vicuna-7B. Verify its license before
# using (TBD-002 in Model_Selection.md). llm_proj is skipped when hidden sizes differ (Vicuna 4096 vs Qwen).
if CFG["init_from_deltavlm_ckpt"]:
    import requests
    ckpt_path = os.path.join(CFG["output_dir"], "deltavlm_checkpoint_best.pth")
    if not os.path.exists(ckpt_path):
        print("downloading", CFG["deltavlm_ckpt"])
        r = requests.get(CFG["deltavlm_ckpt"], stream=True, timeout=1800)
        r.raise_for_status()
        with open(ckpt_path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("model", sd)
    msg = model.load_state_dict(sd, strict=False)
    print("missing keys:", len(msg.missing_keys), "| unexpected keys:", len(msg.unexpected_keys))
    keep = [k for k in msg.missing_keys if not k.startswith("llm_") and "llm_proj" not in k]
    print("non-llm missing (e.g. query_tokens shapes):", keep[:10])
else:
    print("init_from_deltavlm_ckpt is OFF - training from pretrained EVA-ViT-g + BERT-initialized Q-former")
"""))

# ----------------------------------------------------------------------------
CELLS.append(md(
"""## 5. Smoke test (one fwd/bwd + one generation)

Set `CFG["smoke_test"] = True` in the CONFIG cell and run all cells up to here to validate shapes
before the full run. Catches Qwen-unwrap / embedding-size / Q-former mismatches in ~1 minute.
"""))

# ----------------------------------------------------------------------------
CELLS.append(code(
"""if CFG["smoke_test"]:
    from torch.utils.data import DataLoader
    sm = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=0,
                    collate_fn=train_ds.collater)
    batch = next(iter(sm))
    out = model(batch)
    print("loss:", out["loss"].item())
    out["loss"].backward()
    print("backward OK")
    with torch.no_grad():
        gen = model.generate({"image_A": batch["image_A"], "image_B": batch["image_B"],
                              "prompt": batch["text_input"]})
    print("generated:", gen)
    print("SMOKE TEST OK - set smoke_test=False and restart from the CONFIG cell to train.")
    raise SystemExit(0)
print("smoke_test is off - proceeding to training")
"""))

# ----------------------------------------------------------------------------
CELLS.append(md(
"""## 6. Training loop (IMP-035)

AdamW (wd=0.05, no decay on bias/LN) + linear-warmup cosine LR (init 1e-5, warmup 1000 steps - the repo's
schedule), gradient clipping 1.0, bf16 autocast on A100 / fp16+scaler on T4, gradient accumulation to
effective batch 32. **One rolling checkpoint per epoch**: `ckpt_latest/` is written after every epoch and
overwritten (final state after the last epoch). With `push_to_hub=True` (default) it is also **uploaded to
your private HF repo right after each epoch** - same paths every time, so files are overwritten there too,
never duplicated (costs a few minutes of upload per ~3 GB epoch checkpoint). Each checkpoint contains:
`qwen_lora/` (PEFT adapter + tokenizer), `bi_ve_finetuned.pt`, `q_former_finetuned.pt`, `llm_proj.pt`
(including the CSRM layers) - the modular layout expected by IMP-036 registry import.
"""))

# ----------------------------------------------------------------------------
CELLS.append(code(
"""# HF Hub upload helper - called after EVERY epoch checkpoint save (same files overwritten on HF).
from huggingface_hub import HfApi


def get_hf_token():
    token = os.environ.get("HF_TOKEN")
    if token is None:
        try:
            if IN_KAGGLE:
                from kaggle_secrets import UserSecretsClient
                token = UserSecretsClient().get_secret(CFG["hf_token_secret"])
            elif IN_COLAB:
                from google.colab import userdata
                token = userdata.get(CFG["hf_token_secret"])
        except Exception as e:
            print("secret lookup failed:", e)
    return token


def upload_to_hf(tag="", allow_patterns=None):
    \"\"\"Push ckpt_latest to the private HF repo. Same paths each epoch -> files are
    overwritten on HF, never duplicated. Pass lineage.json patterns on the final call.\"\"\"
    if not CFG["push_to_hub"]:
        print("push_to_hub is OFF - skipping upload")
        return False
    token = get_hf_token()
    if not token:
        print("NO HF TOKEN - skipping upload" + (f" ({tag})" if tag else "") +
              ". Add Kaggle secret 'HF_TOKEN' (or Colab userdata) and re-run.")
        return False
    if "<your_hf_username>" in CFG["hf_repo"]:
        print("SKIP upload" + (f" ({tag})" if tag else "") +
              " - edit CFG['hf_repo'] in the CONFIG cell to your HF repo id first.")
        return False
    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=CFG["hf_repo"], repo_type="model", private=True, exist_ok=True)
    except Exception as e:
        print(f"HF repo create failed - skipping upload ({tag}): {e}")
        print("Check: HF_TOKEN belongs to the account in CFG['hf_repo'] and has WRITE permission.")
        return False
    try:
        api.upload_folder(
            repo_id=CFG["hf_repo"], repo_type="model", folder_path=CFG["output_dir"],
            allow_patterns=list(allow_patterns or ["ckpt_latest/**"]),
        )
    except Exception as e:
        print(f"HF upload failed ({tag}): {e}")
        return False
    print(f"uploaded to HF ({tag or 'ckpt_latest'}) -> https://huggingface.co/{CFG['hf_repo']}")
    return True


# probe once so a missing token/repo id fails fast - before hours of training
_probe_token = get_hf_token()
if CFG["push_to_hub"] and _probe_token and "<your_hf_username>" not in CFG["hf_repo"]:
    try:
        HfApi(token=_probe_token).create_repo(
            repo_id=CFG["hf_repo"], repo_type="model", private=True, exist_ok=True)
        print("HF repo ready:", CFG["hf_repo"], "| per-epoch uploads enabled")
    except Exception as e:
        print("WARNING: HF repo create failed -", e)
        print("Uploads will be skipped. Check: HF_TOKEN belongs to the account in CFG['hf_repo']")
        print("(case-sensitive) and has WRITE permission. Fix, then re-run this cell.")
else:
    print("WARNING: HF uploads will be SKIPPED during training. Set the 'HF_TOKEN' secret and")
    print("edit CFG['hf_repo'] in the CONFIG cell to enable them.")
"""))

# ----------------------------------------------------------------------------
CELLS.append(code(
"""# Optional: Download checkpoint from Hugging Face Hub to resume training
if CFG.get("resume_from_hf") and CFG.get("hf_repo"):
    from huggingface_hub import snapshot_download
    token = get_hf_token()
    print(f"Resuming: checking/downloading ckpt_latest from {CFG['hf_repo']} ...")
    try:
        snapshot_download(
            repo_id=CFG["hf_repo"],
            allow_patterns=["ckpt_latest/**"],
            local_dir=CFG["output_dir"],
            token=token,
        )
        CFG["resume_checkpoint_dir"] = os.path.join(CFG["output_dir"], "ckpt_latest")
        print("Resumed checkpoint downloaded successfully to:", CFG["resume_checkpoint_dir"])
        if "model" in globals() and model is not None:
            model.load_checkpoint(CFG["resume_checkpoint_dir"])
    except Exception as e:
        print("HF checkpoint download failed (will train without resume):", e)
elif CFG.get("resume_checkpoint_dir") and os.path.exists(CFG["resume_checkpoint_dir"]):
    print("Resuming from local directory:", CFG["resume_checkpoint_dir"])
    if "model" in globals() and model is not None:
        model.load_checkpoint(CFG["resume_checkpoint_dir"])
"""))

# ----------------------------------------------------------------------------
CELLS.append(code(
"""# Write the self-contained DDP training script (used only when 2+ GPUs are available; the
# single-GPU path below runs the loop inline). Generated from the same model code as the notebook.
_script_text = TRAIN_DDP_SCRIPT_PLACEHOLDER
_script_path = os.path.join(CFG["base_dir"], "train_ddp.py")
with open(_script_path, "w", encoding="utf-8") as _f:
    _f.write(_script_text)
print("train_ddp.py written:", len(_script_text), "chars ->", _script_path)
"""))

# ----------------------------------------------------------------------------
CELLS.append(code(
"""# 2-GPU DDP training (Kaggle '2x T4 GPU'): torchrun with one process per GPU.
# On a single-GPU session this cell is a no-op and the inline training cell below runs instead.
# If the torchrun launch fails, set CFG["use_ddp"]=False in the CONFIG cell and use the inline cell.
import json as _json

WORLD = torch.cuda.device_count() if CFG.get("use_ddp", True) else 1
CFG["world_size"] = WORLD
print("GPUs visible:", WORLD, "| use_ddp:", CFG.get("use_ddp", True))

if WORLD > 1:
    _script_path = os.path.join(CFG["base_dir"], "train_ddp.py")
    _cfg_path = os.path.join(CFG["output_dir"], "train_cfg.json")
    _cfg = {k: CFG[k] for k in (
        "deltavlm_dir", "dataset_dir", "output_dir", "vis_root", "ann_dir",
        "img_size", "num_query_token", "max_txt_len", "max_output_txt_len",
        "lora_r", "lora_alpha", "lora_dropout", "lora_targets",
        "grad_checkpoint", "dtype", "batch_size", "accum_steps", "max_epochs",
        "init_lr", "min_lr", "warmup_steps", "weight_decay", "grad_clip",
        "seed", "num_workers", "data_subset", "llm_backbone", "llm_fallback",
        "save_steps", "resume_checkpoint_dir", "start_step")}
    _cfg["vit_blocks_trainable"] = sorted(CFG["vit_blocks_trainable"])
    _cfg["world_size"] = WORLD
    _cfg["push_to_hub"] = CFG.get("push_to_hub", False)
    _cfg["hf_repo"] = CFG.get("hf_repo", "")
    _cfg["hf_token"] = get_hf_token()
    with open(_cfg_path, "w") as _f:
        _json.dump(_cfg, _f, indent=2)
    print("launching torchrun --nproc_per_node=%d ..." % WORLD)
    # Free in-notebook model completely from RAM & GPU so DDP worker processes
    # have the full ~30 GB host memory available (avoids Kaggle OOM kernel crash).
    if "model" in globals() and model is not None:
        print("freeing in-notebook model from RAM & GPU before launching DDP ranks")
        del model
        model = None
        import gc; gc.collect()
        torch.cuda.empty_cache()
    import subprocess, socket
    # Find a free port for torchrun master_port to avoid "Address already in use"
    _s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _s.bind(("", 0))
    _port = str(_s.getsockname()[1])
    _s.close()

    _env = os.environ.copy()
    # T4 GPUs do not support P2P across PCIe in cloud/virtualized environments;
    # disabling P2P and IB prevents NCCL initialization failure / hanging.
    _env["NCCL_P2P_DISABLE"] = "1"
    _env["NCCL_IB_DISABLE"] = "1"

    _cmd = ["torchrun", "--nproc_per_node", str(WORLD), "--master_port", _port, _script_path, _cfg_path]
    _proc = subprocess.Popen(_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=_env)
    for _line in _proc.stdout:
        print(_line, end="", flush=True)
    _proc.wait()
    if _proc.returncode != 0:
        CFG["ddp_done"] = False
        CFG["use_ddp"] = False  # auto-fallback: the inline cell below trains on GPU 0 instead
        print(f"torchrun exited with code {_proc.returncode} - the 2-GPU path failed.")
        print("AUTO-FALLBACK: rebuilding model on GPU 0 for single-GPU training...")
        model = DeltaVLMQwen(CFG)
        model = model.cuda() if torch.cuda.is_available() else model
        print("Model rebuilt on GPU 0. The next cell will run single-GPU training.")
    else:
        CFG["ddp_done"] = True
        upload_to_hf(tag="ddp final", allow_patterns=["ckpt_latest/**"])
        print("DDP training finished - checkpoints in", CFG["output_dir"])
else:
    print("single GPU - the inline training cell below will run")
"""))

# ----------------------------------------------------------------------------
CELLS.append(code(
"""from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

# Single-GPU inline training. If the DDP cell above already trained on 2 GPUs, skip this.
if CFG.get("ddp_done", False):
    print("DDP training already completed above - skipping this single-GPU cell")
else:
    num_workers = 0 if platform.system() == "Windows" else CFG["num_workers"]
    loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                        num_workers=num_workers, collate_fn=train_ds.collater, drop_last=True)

    steps_per_epoch = len(loader) // CFG["accum_steps"]
    total_steps = CFG["max_epochs"] * steps_per_epoch
    warmup = CFG["warmup_steps"]
    print(f"steps/epoch={steps_per_epoch} (eff bs={CFG['batch_size']*CFG['accum_steps']}) total={total_steps}")

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dtype != torch.float32:
            p.data = p.data.float()
        (no_decay if (p.ndim < 2 or "bias" in n or "norm" in n or "ln" in n) else decay).append(p)
    optimizer = AdamW([{"params": decay, "weight_decay": CFG["weight_decay"]},
                       {"params": no_decay, "weight_decay": 0.0}], lr=CFG["init_lr"])

    def lr_mult(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    scheduler = LambdaLR(optimizer, lr_mult)
    scaler = torch.cuda.amp.GradScaler(enabled=(CFG["dtype"] == "fp16"), init_scale=2**14)

    def save_checkpoint(model, out_dir, tag):
        d = os.path.join(out_dir, tag)
        os.makedirs(d, exist_ok=True)
        model.llm_model.save_pretrained(os.path.join(d, "qwen_lora"))
        model.llm_tokenizer.save_pretrained(os.path.join(d, "qwen_lora"))

        def _half(sd):
            # store fine-tuned weights in fp16 (half the artifact size; inference casts anyway)
            return {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}

        torch.save({"visual_encoder": _half(model.visual_encoder.state_dict()),
                    "ln_vision": _half(model.ln_vision.state_dict())},
                   os.path.join(d, "bi_ve_finetuned.pt"))
        torch.save({"Qformer": _half(model.Qformer.state_dict()),
                    "query_tokens": model.query_tokens.detach().half()},
                   os.path.join(d, "q_former_finetuned.pt"))
        torch.save({"llm_proj": _half(model.llm_proj.state_dict()),
                    "context1": _half(model.context1.state_dict()),
                    "context2": _half(model.context2.state_dict()),
                    "gate1": _half(model.gate1.state_dict()),
                    "gate2": _half(model.gate2.state_dict()),
                    "context3": _half(model.context3.state_dict())},
                   os.path.join(d, "llm_proj.pt"))
        print(f"[save] {tag} -> {d}")

    model.train()
    start_step = CFG.get("start_step", 0)
    for _ in range(start_step):
        scheduler.step()
    opt_step = start_step
    seen = start_step * CFG["accum_steps"]
    skip_batches = seen
    running_loss, t0 = 0.0, time.time()
    pbar = tqdm(total=total_steps, initial=start_step, desc="optimizer steps")
    if start_step > 0:
        print(f"Resuming directly from step {start_step}/{total_steps} (skipping {skip_batches} batches)")
    try:
        for epoch in range(1, CFG["max_epochs"] + 1):
            batch_idx = 0
            for batch in loader:
                batch_idx += 1
                if batch_idx <= skip_batches:
                    continue
                batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                with torch.autocast("cuda", dtype=model.amp_dtype, enabled=torch.cuda.is_available()):
                    loss = model(batch)["loss"] / CFG["accum_steps"]
                scaler.scale(loss).backward()
                running_loss += loss.item() * CFG["accum_steps"]
                seen += 1
                if seen % CFG["accum_steps"] == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], CFG["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    opt_step += 1
                    pbar.update(1)
                    if opt_step % 10 == 0:
                        pbar.set_postfix(loss=f"{running_loss/10:.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                        running_loss = 0.0
                    # Periodic step checkpointing: overwrite ckpt_latest and push to HF
                    if CFG.get("save_steps") and opt_step % CFG["save_steps"] == 0:
                        save_checkpoint(model, CFG["output_dir"], "ckpt_latest")
                        upload_to_hf(tag=f"step {opt_step}", allow_patterns=["ckpt_latest/**"])
            # flush leftover accumulated gradients at the end of the epoch
            if seen % CFG["accum_steps"] != 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], CFG["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1
                pbar.update(1)
            # one rolling checkpoint per epoch - same directory is overwritten each time,
            # then pushed to the private HF repo (same paths there, so also overwritten, not duplicated)
            save_checkpoint(model, CFG["output_dir"], "ckpt_latest")
            upload_to_hf(tag=f"epoch {epoch}")
    except KeyboardInterrupt:
        print("interrupted - partial state saved separately; last good epoch is in ckpt_latest")
        save_checkpoint(model, CFG["output_dir"], "ckpt_interrupted")
        raise

    save_checkpoint(model, CFG["output_dir"], "ckpt_latest")  # final state overwrites the same dir
    print(f"training done in {(time.time()-t0)/3600:.2f} h | final step {opt_step} | checkpoint: ckpt_latest "
          "(already on HF from the last epoch upload; run the Auto-save cell below for lineage.json)")
    CFG["trained_steps"] = opt_step
"""))

# ----------------------------------------------------------------------------
CELLS.append(md(
"""## 7. Gate evaluation (IMP-035) - ChangeChat-105k test splits

- **Binary change classification Acc ≥ 90%** on `changechat_105k_test_binary.json` (1,929 samples),
  greedy generation, answer parsed to yes/no and compared against `changeflag`.
- **Captioning CIDEr ≥ baseline** (+ BLEU-4) on a subset of `changechat_105k_test.json` (5 refs/pair)
  using the repo's vendored scorers. Baseline = DeltaVLM paper number - pin it before declaring the gate
  passed (no tuning). Recorded in the lineage file.
"""))

# ----------------------------------------------------------------------------
CELLS.append(code(
"""# If training ran on 2 GPUs via torchrun, `model` is not bound in this notebook process -
# rebuild it from ckpt_latest for the gate evaluation (this is also the IMP-036 load path).
if "model" not in globals() or model is None:
    from peft import PeftModel
    _ck = os.path.join(CFG["output_dir"], "ckpt_latest")
    model = DeltaVLMQwen(CFG)
    _base = model.llm_model.get_base_model()
    model.llm_model = PeftModel.from_pretrained(_base, os.path.join(_ck, "qwen_lora"))
    _s1 = torch.load(os.path.join(_ck, "bi_ve_finetuned.pt"), map_location="cpu")
    model.visual_encoder.load_state_dict(_s1["visual_encoder"])
    model.ln_vision.load_state_dict(_s1["ln_vision"])
    _s2 = torch.load(os.path.join(_ck, "q_former_finetuned.pt"), map_location="cpu")
    model.Qformer.load_state_dict(_s2["Qformer"])
    model.query_tokens.data.copy_(_s2["query_tokens"])
    _s3 = torch.load(os.path.join(_ck, "llm_proj.pt"), map_location="cpu")
    model.llm_proj.load_state_dict(_s3["llm_proj"])
    for _n in ("context1", "context2", "gate1", "gate2", "context3"):
        getattr(model, _n).load_state_dict(_s3[_n])
    model = model.cuda() if torch.cuda.is_available() else model
    print("rebuilt model from ckpt_latest for evaluation")
else:
    print("using the in-process model for evaluation")
"""))

# ----------------------------------------------------------------------------
CELLS.append(code(
"""# Binary classification Acc - full run (greedy) on ChangeChat-105k test_binary (1,929 samples).
import json
from PIL import Image

bin_ann = json.load(open(os.path.join(CFG["ann_dir"], "changechat_105k_test_binary.json")))
gt_by_id = {r["id"]: r["changeflag"] for r in bin_ann}
ids = list(gt_by_id.keys())[: CFG["eval_binary_max"]]
print("binary eval samples:", len(ids))


def parse_binary(text):
    t = text.strip().lower().replace(".", " ").replace(",", " ")
    toks = t.split()
    if not toks:
        return None
    if toks[0] == "yes" or "yes" in toks[:2]:
        return 1
    if toks[0] == "no" or "no" in toks[:2] or "no difference" in t:
        return 0
    return None


bin_by_id = {r["id"]: r for r in bin_ann}
model.eval()
correct = total = unparsed = 0
fail = []
with torch.no_grad():
    for i in range(0, len(ids), 8):
        chunk = ids[i:i + 8]
        imsA, imsB, prompts = [], [], []
        for c in chunk:
            a, b = bin_by_id[c]["image"]
            pa = Image.open(os.path.join(CFG["vis_root"], a)).convert("RGB")
            pb = Image.open(os.path.join(CFG["vis_root"], b)).convert("RGB")
            imsA.append(eval_vis(pa, pa)[0])
            imsB.append(eval_vis(pb, pb)[0])
            prompts.append(bin_by_id[c]["conversations"][0]["value"].replace("<image>", "").strip())
        outs = model.generate({"image_A": torch.stack(imsA).cuda(),
                               "image_B": torch.stack(imsB).cuda(), "prompt": prompts})
        for c, out in zip(chunk, outs):
            pred = parse_binary(out)
            total += 1
            if pred is None:
                unparsed += 1
                fail.append((c, out))
                continue
            correct += (pred == (1 if gt_by_id[c] != 0 else 0))

acc = correct / max(1, total)
print(f"binary Acc = {correct}/{total} = {acc:.4f}  (unparsed: {unparsed})")
print("unparsed examples:", fail[:5])
gate = acc >= 0.90
print(f"IMP-035 gate (binary Acc >= 90%): {'PASS' if gate else 'NOT YET'}")
CFG["gate_binary_acc"] = round(acc, 4)
CFG["gate_binary_pass"] = bool(gate)
"""))

# ----------------------------------------------------------------------------
CELLS.append(code(
"""# Captioning: CIDEr + BLEU-4 on a subset of changechat_105k_test.json (5 refs per pair).
from eval_func.cider.cider import Cider
from eval_func.bleu.bleu import Bleu

cap_ann = json.load(open(os.path.join(CFG["ann_dir"], "changechat_105k_test.json")))
cap_ann = cap_ann[: CFG["cider_subset"]]
CAPTION_PROMPT = "Please briefly describe the changes in these two images."

gts, res = {}, {}
model.eval()
with torch.no_grad():
    for i in range(0, len(cap_ann), 8):
        chunk = cap_ann[i:i + 8]
        imsA, imsB = [], []
        for r in chunk:
            pa = Image.open(os.path.join(CFG["vis_root"], r["image_A"])).convert("RGB")
            pb = Image.open(os.path.join(CFG["vis_root"], r["image_B"])).convert("RGB")
            imsA.append(eval_vis(pa, pa)[0])
            imsB.append(eval_vis(pb, pb)[0])
        outs = model.generate({"image_A": torch.stack(imsA).cuda(),
                               "image_B": torch.stack(imsB).cuda(),
                               "prompt": [CAPTION_PROMPT] * len(chunk)})
        for r, out in zip(chunk, outs):
            gts[r["id"]] = r["captions"]
            res[r["id"]] = [out]

refs = [gts[k] for k in gts]
hyps = [res[k] for k in res]
cider_score, _ = Cider().compute_score(refs, hyps)
bleu_score, _ = Bleu(4).compute_score(refs, hyps)
print(f"CIDEr = {cider_score:.4f} | BLEU-4 = {bleu_score[-1]:.4f} (n={len(gts)})")
print("baseline (DeltaVLM paper): pin before declaring gate - record, do NOT tune")
CFG["gate_cider"] = round(float(cider_score), 4)
CFG["gate_bleu4"] = round(float(bleu_score[-1]), 4)
"""))

# ----------------------------------------------------------------------------
CELLS.append(code(
"""# Save IMP-036-style lineage for registry import + artifact checksums.
import importlib.metadata as im

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()

artifacts = {}
for tag in sorted(os.listdir(CFG["output_dir"])):
    d = os.path.join(CFG["output_dir"], tag)
    if os.path.isdir(d):
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith((".pt", ".safetensors", ".json")):
                    p = os.path.join(root, f)
                    artifacts[f"{tag}/{os.path.relpath(p, d)}"] = sha256(p)

lineage = {
    "model_key": "deltavlm-qwen",
    "task": "IMP-035 (FT-C)",
    "slot": "S3 change-vqa",
    "architecture": "Bi-VE (EVA-ViT-g/14) + CSRM + Q-former + Qwen3.5-2B (LoRA)",
    "licenses": {
        "deltavlm_code": "Apache-2.0", "changechat_105k": "CC-BY-4.0",
        "qwen3.5_2b": "Apache-2.0", "levir_cc_images": "LEVIR-CC (free, cite Liu et al. 2022)",
    },
    "base_weights": {
        "vision_encoder": "eva_vit_g.pth (sfr-vision-language-research LAVIS)",
        "qformer_init": "bert-base-uncased",
        "llm": CFG["llm_backbone"],
    },
    "deltavlm_commit": CFG.get("deltavlm_commit"),
    "dataset": {
        "annotations": "hlwu/changechat-105k (CC-BY-4.0)",
        "images": "lcybuaa/LEVIR-CC",
        "train_samples": len(train_ds),
    },
    "hyperparams": {k: CFG[k] for k in
                    ["lora_r", "lora_alpha", "lora_dropout", "init_lr", "min_lr", "warmup_steps",
                     "weight_decay", "grad_clip", "batch_size", "accum_steps", "max_epochs",
                     "dtype", "num_query_token", "img_size", "max_txt_len", "max_output_txt_len"]},
    "trainable_groups": {k: round(v / 1e6, 2) for k, v in CFG.get("trainable_groups", {}).items()},
    "env": {p: im.version(p) for p in ["torch", "transformers", "peft", "accelerate", "timm"]},
    "gates": {k: CFG.get(k) for k in ["gate_binary_acc", "gate_binary_pass", "gate_cider", "gate_bleu4"]},
    "artifacts_sha256": artifacts,
}
with open(os.path.join(CFG["output_dir"], "lineage.json"), "w") as f:
    json.dump(lineage, f, indent=2)
print("lineage.json ->", os.path.join(CFG["output_dir"], "lineage.json"))
print("artifact count:", len(artifacts))
"""))

# ----------------------------------------------------------------------------
CELLS.append(code(
"""# Auto-save checkpoints to your computer - same upload as after each epoch (final retry incl. lineage).
if upload_to_hf(tag="final", allow_patterns=["ckpt_latest/**", "lineage.json"]):
    print("download on your computer (IMP-036 import):")
    print(f"  huggingface-cli download {CFG['hf_repo']} --local-dir models/deltavlm-qwen-v2")
else:
    print("Commit the notebook instead, then pull the output locally:")
    print("  kaggle kernels output <owner>/<kernel> -p .")
"""))

# ----------------------------------------------------------------------------
CELLS.append(md(
"""## 8. Next steps

1. **Get the checkpoints onto your computer:** after training, run the "Auto-save via HF Hub" cell -
   it pushes the final checkpoint + `lineage.json` to your private HF repo, then pull locally:

       huggingface-cli download VMamidala/satquery-model-c-deltavlm-qwen --local-dir models/deltavlm-qwen-v2

   (Kaggle alternative if you don't use HF: commit the notebook - the output dataset keeps the files,
   and `kaggle kernels output <owner>/<kernel> -p .` downloads them. The HF route is the automated one.)
2. **IMP-036:** bump `config/registry.json` entry `change-vqa` (model_key `deltavlm-qwen`) to v2 with the
   lineage pins + checksums; artifacts land under `models/` (gitignored).
3. **IMP-037 (separate):** dry-run this adapted checkpoint on CDVQA test1 + test2 - measurement only,
   no tuning. Not part of this notebook by design (DEC-021).
4. **Kaggle T4:** the CONFIG cell auto-applies fp16 / batch 2 / accum 8 / grad-checkpoint when it detects
   a Kaggle runtime; expect ~2-3x longer wall time than an A100.
5. **Full gate:** binary Acc ≥90% needs the full ~8.2k-step run; a shorter run is a diagnostic only.

**Per-run README:** `training/model_c/README.md` in the repo.
"""))

# ----------------------------------------------------------------------------
# Compose the self-contained DDP training script (used when >1 GPU is visible).
# The notebook's own model-cell source (helpers + DeltaVLMQwen class) is reused verbatim,
# so the DDP run and the single-GPU run can never drift apart.

DDP_PRELUDE = '''# Self-contained DDP training script for Model C (generated by the notebook).
# Usage: torchrun --nproc_per_node=N train_ddp.py train_cfg.json
import os, sys, json, math, time, random, platform
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

CFG = json.load(open(sys.argv[1]))
CFG["vit_blocks_trainable"] = set(CFG["vit_blocks_trainable"])
WORLD = CFG["world_size"]

sys.path.insert(0, CFG["deltavlm_dir"])
sys.path.insert(0, os.path.dirname(CFG["deltavlm_dir"]))
os.chdir(CFG["deltavlm_dir"])

# ---- timm shim (same as the notebook) ----
import types
try:
    import timm.models.hub as _hub  # noqa: F401
except Exception:
    from urllib.parse import urlparse
    import requests

    def _dcf(url, check_hash=True, progress=False):
        dst = os.path.join(os.path.expanduser("~"), ".cache", "timm", os.path.basename(urlparse(url).path))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            r = requests.get(url, stream=True, timeout=600)
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        return dst

    _m = types.ModuleType("timm.models.hub")
    _m.get_cache_dir = lambda: os.path.join(os.path.expanduser("~"), ".cache", "timm")
    _m.download_cached_file = _dcf
    sys.modules["timm.models.hub"] = _m
if "timm.models.layers" not in sys.modules:
    try:
        from timm.models.layers import drop_path, to_2tuple, trunc_normal_  # noqa: F401
    except Exception:
        from timm.layers import drop_path, to_2tuple, trunc_normal_  # noqa: F401
        _ml = types.ModuleType("timm.models.layers")
        _ml.drop_path, _ml.to_2tuple, _ml.trunc_normal_ = drop_path, to_2tuple, trunc_normal_
        sys.modules["timm.models.layers"] = _ml

# ---- transformers v5 compat shims (same as the notebook) ----
import transformers.modeling_utils as _mu


def _inject(name, fn):
    if not hasattr(_mu, name):
        setattr(_mu, name, fn)


_inject("apply_chunking_to_forward", lambda fn, chunk_size, chunk_dim, *t: fn(*t))


def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head -= sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index


def _prune_linear_layer(layer, index, dim=0):
    index = index.to(layer.weight.device)
    if dim == 0:
        new = nn.Linear(layer.in_features, len(index), bias=layer.bias is not None)
        new.weight.data = layer.weight.data[index, :].clone()
        if layer.bias is not None:
            new.bias.data = layer.bias.data[index].clone()
    elif dim == 1:
        new = nn.Linear(len(index), layer.out_features, bias=layer.bias is not None)
        new.weight.data = layer.weight.data[:, index].clone()
        if layer.bias is not None:
            new.bias.data = layer.bias.data.clone()
    new.weight.requires_grad_(False)
    if layer.bias is not None:
        new.bias.requires_grad_(False)
    return new


_inject("find_pruneable_heads_and_indices", _find_pruneable_heads_and_indices)
_inject("prune_linear_layer", _prune_linear_layer)

from model import blip2 as _blip2  # noqa: E402
from model.Qformer import (  # noqa: E402
    BertConfig as _BertConfig,
    BertPreTrainedModel as _BertPreTrainedModel,
    BertLMHeadModel as _BertLMHeadModel,
)

_BertPreTrainedModel.all_tied_weights_keys = {}
_BertPreTrainedModel._tied_weights_keys = {}
_BertPreTrainedModel.tie_weights = lambda self, *a, **k: None


def _add_method(cls, name, fn):
    if not hasattr(cls, name):
        setattr(cls, name, fn)


def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
    if head_mask is not None:
        head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked:
            head_mask = head_mask.unsqueeze(-1)
    else:
        head_mask = [None] * num_hidden_layers
    return head_mask


def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    assert head_mask.dim() == 5, f"head_mask.dim() != 5, instead {head_mask.dim()}"
    head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
    return head_mask


def _invert_attention_mask(self, encoder_attention_mask):
    if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    elif encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    else:
        raise ValueError(f"wrong shape for encoder_attention_mask: {encoder_attention_mask.shape}")
    dtype = next(self.parameters()).dtype
    encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=dtype)
    if getattr(self.config, "is_decoder", False):
        device = encoder_attention_mask.device
        batch_size, seq_length = encoder_attention_mask.shape
        seq_ids = torch.arange(seq_length, device=device)
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        causal_mask = causal_mask.to(dtype)
        if causal_mask.shape[1] < encoder_extended_attention_mask.shape[1]:
            prefix = encoder_extended_attention_mask.shape[1] - causal_mask.shape[1]
            causal_mask = torch.cat(
                [torch.ones((batch_size, seq_length, prefix), device=device, dtype=dtype), causal_mask],
                dim=-1)
        encoder_extended_attention_mask = encoder_extended_attention_mask * causal_mask[:, None, :, :]
    encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(dtype).min
    return encoder_extended_attention_mask


_add_method(_BertPreTrainedModel, "get_head_mask", _get_head_mask)
_add_method(_BertPreTrainedModel, "_convert_head_mask_to_5d", _convert_head_mask_to_5d)
_add_method(_BertPreTrainedModel, "invert_attention_mask", _invert_attention_mask)


def _patched_init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
    encoder_config = _BertConfig.from_pretrained("../bert-base-uncased")
    encoder_config.encoder_width = vision_width
    encoder_config.add_cross_attention = True
    encoder_config.cross_attention_freq = cross_attention_freq
    encoder_config.query_length = num_query_token
    Qformer = _BertLMHeadModel(encoder_config)
    bert_sd = torch.load(os.path.join("..", "bert-base-uncased", "pytorch_model.bin"), map_location="cpu")
    missing, unexpected = Qformer.load_state_dict(bert_sd, strict=False)
    print(f"[compat] Qformer built from config + BERT weights: {len(missing)} missing, {len(unexpected)} unexpected", flush=True)
    query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
    query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    return Qformer, query_tokens


_blip2.Blip2Base.init_Qformer = classmethod(_patched_init_Qformer)
print("DDP prelude ready", flush=True)
'''

DDP_TRAIN = '''
# ---- dataset (built per rank) ----
from processor import Blip2ImageTrainProcessor, BlipCaptionProcessor
from dataset import CaptionDataset
from huggingface_hub import HfApi


def upload_to_hf(tag="", allow_patterns=None):
    if not CFG.get("push_to_hub") or not CFG.get("hf_token"):
        return False
    if "<your_hf_username>" in CFG.get("hf_repo", ""):
        return False
    try:
        api = HfApi(token=CFG["hf_token"])
        api.create_repo(repo_id=CFG["hf_repo"], repo_type="model", private=True, exist_ok=True)
        api.upload_folder(
            repo_id=CFG["hf_repo"], repo_type="model", folder_path=CFG["output_dir"],
            allow_patterns=list(allow_patterns or ["ckpt_latest/**"]),
        )
        print(f"[rank0] uploaded {tag} to HF -> https://huggingface.co/{CFG['hf_repo']}", flush=True)
        return True
    except Exception as e:
        print(f"[rank0] HF upload failed ({tag}): {e}", flush=True)
        return False

train_vis = Blip2ImageTrainProcessor(image_size=CFG["img_size"])
txt_proc = BlipCaptionProcessor()
train_path = os.path.join(CFG["ann_dir"], "changechat_105k_train.json")
train_ds = CaptionDataset(vis_processor=train_vis, text_processor=txt_proc,
                          vis_root=CFG["vis_root"], ann_paths=[train_path])
if CFG.get("data_subset"):
    train_ds.annotation = train_ds.annotation[: CFG["data_subset"]]


def main_rank(local_rank):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    random.seed(CFG["seed"] + local_rank)
    np.random.seed(CFG["seed"] + local_rank)
    torch.manual_seed(CFG["seed"] + local_rank)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model = DeltaVLMQwen(CFG)
    if CFG.get("resume_checkpoint_dir") and os.path.exists(CFG["resume_checkpoint_dir"]):
        model.load_checkpoint(CFG["resume_checkpoint_dir"])
    model = model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], find_unused_parameters=False)

    sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds, num_replicas=WORLD, rank=local_rank, shuffle=True, drop_last=True)
    num_workers = 0 if platform.system() == "Windows" else CFG["num_workers"]
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=CFG["batch_size"], num_workers=num_workers,
        collate_fn=train_ds.collater, sampler=sampler, drop_last=True)

    steps_per_epoch = len(loader) // CFG["accum_steps"]
    total_steps = CFG["max_epochs"] * steps_per_epoch
    warmup = CFG["warmup_steps"]
    eff_bs = CFG["batch_size"] * CFG["accum_steps"] * WORLD
    if local_rank == 0:
        print(f"[rank0] per-rank batch {CFG['batch_size']} x accum {CFG['accum_steps']} x {WORLD} GPU "
              f"= effective bs {eff_bs} | steps/epoch {steps_per_epoch} | total {total_steps}", flush=True)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dtype != torch.float32:
            p.data = p.data.float()
        (no_decay if (p.ndim < 2 or "bias" in n or "norm" in n or "ln" in n) else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": CFG["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}], lr=CFG["init_lr"])

    def lr_mult(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_mult)
    scaler = torch.cuda.amp.GradScaler(enabled=(CFG["dtype"] == "fp16"), init_scale=2**14)
    dt = torch.bfloat16 if CFG["dtype"] == "bf16" else torch.float16

    def save_checkpoint(m, out_dir, tag):
        d = os.path.join(out_dir, tag)
        os.makedirs(d, exist_ok=True)
        m.llm_model.save_pretrained(os.path.join(d, "qwen_lora"))
        m.llm_tokenizer.save_pretrained(os.path.join(d, "qwen_lora"))

        def _half(sd):
            return {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}

        torch.save({"visual_encoder": _half(m.visual_encoder.state_dict()),
                    "ln_vision": _half(m.ln_vision.state_dict())},
                   os.path.join(d, "bi_ve_finetuned.pt"))
        torch.save({"Qformer": _half(m.Qformer.state_dict()),
                    "query_tokens": m.query_tokens.detach().half()},
                   os.path.join(d, "q_former_finetuned.pt"))
        torch.save({"llm_proj": _half(m.llm_proj.state_dict()),
                    "context1": _half(m.context1.state_dict()),
                    "context2": _half(m.context2.state_dict()),
                    "gate1": _half(m.gate1.state_dict()),
                    "gate2": _half(m.gate2.state_dict()),
                    "context3": _half(m.context3.state_dict())},
                   os.path.join(d, "llm_proj.pt"))
        if local_rank == 0:
            print(f"[rank0] saved {tag} -> {d}", flush=True)

    model.train()
    start_step = CFG.get("start_step", 0)
    for _ in range(start_step):
        scheduler.step()
    opt_step = start_step
    seen = start_step * CFG["accum_steps"]
    skip_batches = seen
    running_loss, t0 = 0.0, time.time()
    if local_rank == 0 and start_step > 0:
        print(f"[rank0] Resuming directly from step {start_step}/{total_steps} (skipping {skip_batches} batches)", flush=True)
    try:
        for epoch in range(1, CFG["max_epochs"] + 1):
            sampler.set_epoch(epoch)
            batch_idx = 0
            for batch in loader:
                batch_idx += 1
                if batch_idx <= skip_batches:
                    continue
                batch = {k: (v.cuda(local_rank) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}
                with torch.autocast("cuda", dtype=dt):
                    loss = model(batch)["loss"] / CFG["accum_steps"]
                scaler.scale(loss).backward()
                running_loss += loss.item() * CFG["accum_steps"]
                seen += 1
                if seen % CFG["accum_steps"] == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], CFG["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    opt_step += 1
                    if local_rank == 0 and opt_step % 10 == 0:
                        _el = time.time() - t0
                        _eta = _el / max(1, opt_step) * (total_steps - opt_step)
                        print(f"[rank0] step {opt_step}/{total_steps} ({100.0*opt_step/total_steps:4.1f}%) "
                              f"loss {running_loss/10:.3f} lr {scheduler.get_last_lr()[0]:.2e} "
                              f"eta {_eta/3600:.2f}h", flush=True)
                        running_loss = 0.0
                    # Periodic step checkpointing: overwrite ckpt_latest and push to HF (rank 0 only)
                    if local_rank == 0 and CFG.get("save_steps") and opt_step % CFG["save_steps"] == 0:
                        save_checkpoint(model.module, CFG["output_dir"], "ckpt_latest")
                        upload_to_hf(tag=f"step {opt_step}", allow_patterns=["ckpt_latest/**"])
            # flush leftover accumulated gradients at the end of the epoch
            if seen % CFG["accum_steps"] != 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], CFG["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1
            # one rolling checkpoint per epoch, overwritten (rank 0 only)
            if local_rank == 0:
                save_checkpoint(model.module, CFG["output_dir"], "ckpt_latest")
                upload_to_hf(tag=f"epoch {epoch}")
            dist.barrier()
    except KeyboardInterrupt:
        if local_rank == 0:
            save_checkpoint(model.module, CFG["output_dir"], "ckpt_interrupted")
        raise
    if local_rank == 0:
        save_checkpoint(model.module, CFG["output_dir"], "ckpt_latest")
        upload_to_hf(tag="final")
        print(f"[rank0] training done in {(time.time()-t0)/3600:.2f} h | final step {opt_step}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main_rank(int(os.environ.get("LOCAL_RANK", 0)))
'''

_model_src = [c["source"] for c in CELLS if "class DeltaVLMQwen" in c["source"]][0]
_script = DDP_PRELUDE + "\n\n" + _model_src + "\n\n" + DDP_TRAIN
for _c in CELLS:
    if "TRAIN_DDP_SCRIPT_PLACEHOLDER" in _c["source"]:
        _c["source"] = _c["source"].replace("TRAIN_DDP_SCRIPT_PLACEHOLDER", json.dumps(_script))

# ----------------------------------------------------------------------------
NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)

print(f"wrote {OUT} | {len(CELLS)} cells | {os.path.getsize(OUT)/1024:.1f} KB")
