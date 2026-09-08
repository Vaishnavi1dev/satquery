"""
Kaggle Fine-Tuning Script: Model C (DeltaVLM + Qwen3.5-2B)
Architecture: Bi-VE (EVA-ViT-g/14) + IDPM (Q-former) + Qwen3.5-2B
Task: LoRA on Qwen3.5-2B, selective FT on Bi-VE last 2 blocks, FT Q-former on ChangeChat-105k.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import argparse

# -----------------------------------------------------------------------------
# 1. Mock Architecture Components (Replace with DeltaVLM's actual codebase)
# -----------------------------------------------------------------------------

class DeltaVLMBiVE(nn.Module):
    """Bi-temporal Vision Encoder (EVA-ViT-g/14)."""
    def __init__(self):
        super().__init__()
        # TODO: Load actual EVA-ViT-g/14 
        # For memory efficiency, usually loaded in Int8 or bf16
        self.blocks = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(40)]) # Mock blocks
        
    def forward(self, img_t1, img_t2):
        return torch.randn(img_t1.shape[0], 196, 1024).to(img_t1.device)

class IDPMQformer(nn.Module):
    """Instruction-guided Difference Perception Module."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1024, 2048) # Project to LLM hidden size
        
    def forward(self, bi_feats, instructions):
        return self.proj(bi_feats)

# -----------------------------------------------------------------------------
# 2. Dataset Definition (ChangeChat-105k)
# -----------------------------------------------------------------------------

class ChangeChatDataset(Dataset):
    """Loads bi-temporal image pairs and instructions from ChangeChat-105k."""
    def __init__(self, data_dir, split="train"):
        self.data_dir = data_dir
        self.samples = [{"t1": "t1.jpg", "t2": "t2.jpg", "instruction": "What changed?", "response": "New buildings."}] * 100

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.randn(3, 224, 224), torch.randn(3, 224, 224), self.samples[idx]["instruction"], self.samples[idx]["response"]

# -----------------------------------------------------------------------------
# 3. Model Setup & Freezing (IMP-035)
# -----------------------------------------------------------------------------

def setup_model(device):
    # 1. LLM (Qwen3.5-2B) with LoRA
    llm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-1.8B", # Using 1.5-1.8B as placeholder for 3.5-2B if offline, adjust accordingly
        torch_dtype=torch.bfloat16,
        device_map="auto" 
    )
    
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    llm = get_peft_model(llm, lora_config)
    llm.print_trainable_parameters()
    
    # 2. Vision Encoder (Bi-VE)
    bi_ve = DeltaVLMBiVE().to(device)
    # Freeze all except last 2 blocks
    for param in bi_ve.parameters():
        param.requires_grad = False
    for block in bi_ve.blocks[-2:]:
        for param in block.parameters():
            param.requires_grad = True
            
    # 3. IDPM / Q-Former (Fully trainable)
    q_former = IDPMQformer().to(device)
    for param in q_former.parameters():
        param.requires_grad = True
        
    return bi_ve, q_former, llm

# -----------------------------------------------------------------------------
# 4. Training Loop
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/kaggle/input/changechat-105k")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Setting up Model C on {device}")
    
    bi_ve, q_former, llm = setup_model(device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B")

    dataset = ChangeChatDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer for all trainable params
    trainable_params = list(llm.parameters()) + list(bi_ve.blocks[-2:].parameters()) + list(q_former.parameters())
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, trainable_params), lr=2e-5)

    print("Starting Training on ChangeChat-105k...")
    for epoch in range(args.epochs):
        for batch_idx, (img_t1, img_t2, inst, resp) in enumerate(dataloader):
            img_t1, img_t2 = img_t1.to(device), img_t2.to(device)
            
            optimizer.zero_grad()
            
            # Forward Vision
            bi_feats = bi_ve(img_t1, img_t2)
            fusion_feats = q_former(bi_feats, inst)
            
            # Tokenize & Forward LLM (Placeholder logic)
            inputs = tokenizer(inst, return_tensors="pt", padding=True).to(device)
            labels = tokenizer(resp, return_tensors="pt", padding=True).input_ids.to(device)
            
            # TODO: Integrate fusion_feats as input embeddings to the LLM
            # outputs = llm(inputs_embeds=fusion_feats, labels=labels)
            
            loss = torch.tensor(0.5, requires_grad=True).to(device) # Mock loss
            loss.backward()
            optimizer.step()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    # Save Adapters and Checkpoints
    output_dir = "/kaggle/working/deltavlm_qwen_adapted"
    os.makedirs(output_dir, exist_ok=True)
    
    llm.save_pretrained(os.path.join(output_dir, "qwen_lora"))
    torch.save(bi_ve.state_dict(), os.path.join(output_dir, "bi_ve_finetuned.pt"))
    torch.save(q_former.state_dict(), os.path.join(output_dir, "q_former_finetuned.pt"))
    print(f"Checkpoints saved to {output_dir}")

if __name__ == "__main__":
    main()
