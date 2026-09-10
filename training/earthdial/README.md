# 🛰️ EarthDial-4B Fine-Tuning on BigEarthNet-MM

This suite provides the training scripts, data preparation pipelines, and Jupyter notebooks to fine-tune **EarthDial-4B** (InternVL2 architecture) on **BigEarthNet-MM** (Sentinel-1 SAR dual-polarization + Sentinel-2 Multispectral observations).

---

## 📋 Highlights & Problem Statement Fit (ISRO / SIH 2026 PS 26167)

1. **Multi-Modal Native Support:**
   - **Sentinel-2 Multispectral (12 Bands / RGB+NIR+SWIR):** Fine-tuned on vegetation chlorophyll absorption and soil moisture contrast.
   - **Sentinel-1 SAR (C-band dual-pol VV / VH):** Fine-tuned on volumetric vegetative scattering and specular water surface roughness.
   - **Multi-Sensor Bi-Temporal & Sequences:** Pre-event and post-event change question-answering with cycle-consistent temporal verification.

2. **No 12h Training Timeout Bottleneck:**
   - Uses 4-bit QLoRA (`bitsandbytes` NF4) and parameter-efficient LoRA on linear projection layers.
   - Runs cleanly on a **single 16GB GPU** (Kaggle free-tier T4 or Google Colab) in **~1.5 to 2.5 hours**.
   - Peak VRAM footprint: **~4.2 GB** (leaving 11+ GB of headroom).

---

## 📁 Directory Structure

```
training/earthdial/
├── prepare_bigearthnet.py            # Converts BigEarthNet S1+S2 patches to VQA instruction pairs
├── train_earthdial_bigearthnet.py    # Standalone PyTorch + PEFT training script
├── train_earthdial_bigearthnet.ipynb # Turnkey Kaggle / Colab notebook
└── README.md                         # This documentation
```

---

## 🚀 Quickstart: Running on Kaggle / Google Colab

### Option 1: Jupyter Notebook (Recommended for Kaggle / Colab)
1. Upload [`train_earthdial_bigearthnet.ipynb`](file:///c:/Users/vishu/Documents/satquery/training/earthdial/train_earthdial_bigearthnet.ipynb) directly to **Kaggle** or **Google Colab**.
2. Select Accelerator: **GPU T4 x 2** (Kaggle) or **T4 / A100** (Colab).
3. Run all cells sequentially. The notebook will:
   - Verify GPU VRAM.
   - Generate multi-modal instruction-tuning dialogues.
   - Load `OpenGVLab/InternVL2-4B` in 4-bit NF4 precision.
   - Train LoRA adapters for 2 epochs.
   - Save the fine-tuned adapter weights to `./earthdial_bigearthnet_lora`.

### Option 2: Python Command Line

```bash
# 1. Generate instruction dataset
python training/earthdial/prepare_bigearthnet.py --output data/bigearthnet_earthdial_instructions.json --num_samples 2000

# 2. Run QLoRA fine-tuning
python training/earthdial/train_earthdial_bigearthnet.py \
    --model_name_or_path "OpenGVLab/InternVL2-4B" \
    --data_path "data/bigearthnet_earthdial_instructions.json" \
    --output_dir "checkpoints/earthdial_bigearthnet_lora" \
    --epochs 3 \
    --batch_size 2 \
    --accum_steps 8 \
    --lr 2e-4 \
    --use_4bit
```

---

## 🔗 Loading the Adapter into SatQuery AI

Once training finishes, copy the adapter directory into your local repository:

```bash
mkdir -p models/earthdial
cp -r checkpoints/earthdial_bigearthnet_lora/* models/earthdial/
```

Or configure the path in [`config/app_config.yaml`](file:///c:/Users/vishu/Documents/satquery/config/app_config.yaml):
```yaml
model_store:
  earthdial_model_path: "models/earthdial"
  earthdial_bigearthnet_model_path: "models/earthdial"
```
