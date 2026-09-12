# 🛰️ Model B: DOFA ViT-B + Cross-Modal Attention Fusion Head
### Slot S4: `opt-sar-fusion` (Co-Registered Optical / Multi-Spectral + SAR Fusion)

This training suite provides data preparation pipelines, production PyTorch training scripts, and turnkey Jupyter notebooks to fine-tune **Model B** for cross-sensor synergistic analysis across Sentinel-2 optical reflectance and Sentinel-1 SAR microwave radar observations.

---

## ⚡ Architecture & Mathematical Foundation

```
[Optical Image: Sentinel-2 B2,B3,B4,B8] ──> [Frozen DOFA ViT-B] ──> Optical Tokens [B, 196, 768] (Query Q)
                                                                            │
                                                                   [Cross-Attention] ──> Fused Tokens [B, 196, 768]
                                                                            │                     │
[SAR Image: Sentinel-1 C-band VV/VH]     ──> [Frozen DOFA ViT-B] ──> SAR Tokens [B, 196, 768]    (Projection Head)
                                                                       (Key K, Value V)           │
                                                                                          [LLM Token Logits]
```

1. **Wavelength-Conditioned Vision Backbone (DOFA ViT-B):**
   - Base weights: `earthflow/DOFA-ViT-B` (~86M parameters).
   - Generates patch embedding weights dynamically using Fourier harmonic projections of band center wavelengths ($\lambda = [0.490, 0.560, 0.665, 0.842]\,\mu\text{m}$ for Sentinel-2, $\lambda = 56,000\,\mu\text{m}$ for Sentinel-1 C-band).
   - **100% Frozen:** Preserves pre-trained multi-sensor representations without catastrophic forgetting (IMP-034 / DEC-004).

2. **Trainable Multi-Head Cross-Attention Head:**
   - Multi-Head Cross-Attention where Optical features act as queries ($Q$), and SAR backscatter features act as keys ($K$) and values ($V$).
   - Dual-directional attention and residual layer normalization:
     $$\text{Opt}_{\text{enhanced}} = \text{LayerNorm}(\text{Opt} + \text{MultiHead}(Q_{\text{opt}}, K_{\text{sar}}, V_{\text{sar}}))$$
   - Multi-layer perceptron (MLP) non-linear projection aligns the fused representations with the text decoder.
   - **Lightweight:** Only **~12M trainable parameters**.

3. **Ultra-Low Compute Footprint:**
   - Peak VRAM during training is **< 4.0 GB VRAM**.
   - Runs easily on free Google Colab (Tesla T4) in **~1 to 1.5 hours**.

---

## 📁 Directory Structure

```
training/dofa/
├── prepare_dofa_pairs.py    # Ingests and prepares Optical + SAR co-registered pairs
├── train_dofa_fusion.py     # Production PyTorch training script (DDP + Checkpointing)
├── train_dofa_fusion.ipynb  # Turnkey Google Colab / Kaggle notebook
└── README.md                # This documentation
```

---

## 🚀 How to Run

### 1. In Google Colab (Turnkey)

1. Open [`training/dofa/train_dofa_fusion.ipynb`](file:///c:/Users/vishu/Documents/satquery/training/dofa/train_dofa_fusion.ipynb) in Google Colab.
2. Select **Runtime** > **Change runtime type** > **T4 GPU**.
3. Add your `HF_TOKEN` in the Secrets tab.
4. Run all cells. Checkpoints automatically sync to `VMamidala/satquery-model-b-dofa-fusion`.

### 2. Single GPU Terminal Execution

```bash
# Step 1: Ingest and prepare Optical-SAR pairs
python training/dofa/prepare_dofa_pairs.py \
    --output data/dofa_fusion_instructions.json \
    --image_dir data/dofa_pairs \
    --num_samples 5000

# Step 2: Train the Cross-Modal Fusion Head
python training/dofa/train_dofa_fusion.py \
    --data_path data/dofa_fusion_instructions.json \
    --output_dir checkpoints/dofa_fusion_head \
    --epochs 3 \
    --batch_size 4 \
    --accum_steps 4 \
    --lr 1e-4 \
    --save_steps 50 \
    --push_to_hub \
    --hf_repo "VMamidala/satquery-model-b-dofa-fusion"
```

### 3. Multi-GPU Distributed Data Parallel (DDP)

On multi-GPU nodes or Kaggle (2x T4 GPUs):
```bash
torchrun --nproc_per_node=2 training/dofa/train_dofa_fusion.py \
    --data_path data/dofa_fusion_instructions.json \
    --output_dir checkpoints/dofa_fusion_head \
    --batch_size 4 \
    --accum_steps 2 \
    --epochs 3
```

### 4. Resuming from a Rolling Checkpoint

If a run is preempted, resume immediately:
```bash
python training/dofa/train_dofa_fusion.py \
    --resume_from_checkpoint checkpoints/dofa_fusion_head/ckpt_latest
```

---

## 🔗 SatQuery Runtime Integration

Copy the fine-tuned fusion head weights into SatQuery's model store:
```bash
mkdir -p models/dofa
cp checkpoints/dofa_fusion_head/ckpt_final/fusion_head.pt models/dofa/
```

When present, `OpticalSARFusionTool` (`app/tools/opt_sar_fusion.py`) directly invokes the fine-tuned cross-modal fusion head to resolve complex dual-sensor queries (e.g. penetrating cloud/haze or distinguishing high-reflectance roofs from radar double-bounce urban structures).
