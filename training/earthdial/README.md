# 🛰️ EarthDial-4B Fine-Tuning on BigEarthNet-MM
### Multi-GPU DDP + Gradient Checkpointing + Step-Wise Checkpointing

This suite provides the training scripts, data preparation pipelines, and Jupyter notebooks to fine-tune **EarthDial-4B** (InternVL2 architecture) on **BigEarthNet-MM** (Sentinel-1 SAR dual-polarization + Sentinel-2 Multispectral observations).

---

## ⚡ Performance, DDP & Checkpointing Features

1. **Multi-GPU DistributedDataParallel (DDP):**
   - Seamlessly scales across multiple GPUs (e.g. **Kaggle 2x T4 GPUs** or multi-GPU A100/V100 nodes) via `torchrun`.
   - Utilizes `torch.distributed` with `nccl` backend, `DistributedSampler` for balanced data distribution, and barrier synchronization.
   - Effective batch size: $\text{batch\_size} \times \text{accum\_steps} \times \text{WORLD\_SIZE}$.

2. **Gradient Checkpointing:**
   - Enabled via `--grad_checkpoint` with `use_reentrant=False` and `enable_input_require_grads()`.
   - Halves activation memory footprint so 4B VLM fits into **<4.5 GB VRAM per GPU**, leaving plenty of headroom.

3. **Step-Wise Rolling Checkpoints & Safe Resume:**
   - **Rolling Checkpoint:** Automatically updates `ckpt_latest/` every $N$ steps (`--save_steps 50`).
   - **Full State Preservation:** Saves adapter weights, optimizer states, learning rate scheduler state, current step, and epoch in `training_state.pt`.
   - **Preemption Resilience:** Resume anytime using `--resume_from_checkpoint checkpoints/earthdial_bigearthnet_lora/ckpt_latest`.
   - **Emergency Checkpoint:** Traps `KeyboardInterrupt` to dump `ckpt_interrupted/` before exiting.

---

## 📁 Directory Structure

```
training/earthdial/
├── prepare_bigearthnet.py            # Converts BigEarthNet S1+S2 patches to VQA instruction pairs
├── train_earthdial_bigearthnet.py    # DDP + Checkpointing PyTorch training script
├── train_earthdial_bigearthnet.ipynb # Turnkey Kaggle / Colab notebook
└── README.md                         # This documentation
```

---

## 🚀 How to Run

### 1. Multi-GPU DDP on Kaggle (2x T4 GPUs)

In Kaggle notebook or terminal (with GPU T4 x 2 selected):

```bash
# Step 1: Prepare BigEarthNet-MM instruction dataset
python training/earthdial/prepare_bigearthnet.py \
    --output data/bigearthnet_earthdial_instructions.json \
    --num_samples 3000

# Step 2: Launch 2-GPU DDP training with torchrun
torchrun --nproc_per_node=2 training/earthdial/train_earthdial_bigearthnet.py \
    --model_name_or_path "OpenGVLab/InternVL2-4B" \
    --data_path "data/bigearthnet_earthdial_instructions.json" \
    --output_dir "checkpoints/earthdial_bigearthnet_lora" \
    --epochs 3 \
    --batch_size 2 \
    --accum_steps 4 \
    --lr 2e-4 \
    --save_steps 50 \
    --use_4bit \
    --grad_checkpoint
```

### 2. Single-GPU (Google Colab / Local Cloud GPU)

```bash
python training/earthdial/train_earthdial_bigearthnet.py \
    --model_name_or_path "OpenGVLab/InternVL2-4B" \
    --data_path "data/bigearthnet_earthdial_instructions.json" \
    --output_dir "checkpoints/earthdial_bigearthnet_lora" \
    --epochs 3 \
    --batch_size 2 \
    --accum_steps 8 \
    --save_steps 50 \
    --use_4bit
```

### 3. Resuming from a Saved Checkpoint

If a session is interrupted or preempted, simply pass `--resume_from_checkpoint`:

```bash
python training/earthdial/train_earthdial_bigearthnet.py \
    --resume_from_checkpoint "checkpoints/earthdial_bigearthnet_lora/ckpt_latest"
```

---

## 🔗 Integrating the Fine-Tuned Adapter into SatQuery AI

Copy the fine-tuned checkpoint (`ckpt_latest` or `ckpt_final`) into your SatQuery weights directory:

```bash
mkdir -p models/earthdial
cp -r checkpoints/earthdial_bigearthnet_lora/ckpt_final/* models/earthdial/
```

Then in [`config/app_config.yaml`](file:///c:/Users/vishu/Documents/satquery/config/app_config.yaml):
```yaml
model_store:
  earthdial_model_path: "models/earthdial"
  earthdial_bigearthnet_model_path: "models/earthdial"
```
