import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import optim
from torch.utils.data import DataLoader, Dataset
from typing import List
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from .config import MAX_GEN_TOKENS, MIN_GEN_TOKENS, SFT_EPOCHS, SFT_BATCH, SFT_LR, GEN_MODEL

DEVICE = "cuda"

# -------------------------
# Load Generator + Tokenizer
# -------------------------
def load_generator(model_name=GEN_MODEL, device=DEVICE):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.eval()
    return tokenizer, model


# -------------------------
# Prompt Construction
# -------------------------
def build_rag_prompt(question: str, retrieved_docs: List[str]) -> str:
    prompt = "### Context:\n"
    for i, doc in enumerate(retrieved_docs, 1):
        prompt += f"[{i}] {doc}\n"
    prompt += "\n### Question:\n" + question + "\n### Answer:\n"
    return prompt


# -------------------------
# Text Generation (Inference)
# -------------------------
def generate_answer(
    generator,
    tokenizer,
    prompt: str,
    max_new_tokens=MAX_GEN_TOKENS,
    min_new_tokens=MIN_GEN_TOKENS,
    device=DEVICE,
    num_return_sequences=1,
    temperature=0.8,
):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    out = generator.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        do_sample=True,
        top_k=50,
        temperature=temperature,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
    )
    texts = [
        tokenizer.decode(seq[inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        for seq in out
    ]
    return texts


# -------------------------
# LoRA-based Fine-tuning (PEFT)
# -------------------------
class SeqDataset(Dataset):
    def __init__(self, encodings, pad_token_id):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.pad_token_id = pad_token_id

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        ids = self.input_ids[idx]
        mask = self.attention_mask[idx]
        labels = ids.clone()
        labels[labels == self.pad_token_id] = -100  # mask padding for loss
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def sft_finetune_generator(
    generator,
    tokenizer,
    qa_pairs,
    device=DEVICE,
    epochs=SFT_EPOCHS,
    batch_size=SFT_BATCH,
    lr=SFT_LR,
    lora_r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=None,
    save_adapter_path="./saved_models_improved/lora_adapter",
):
    """
    PEFT + LoRA fine-tuning instead of full model training.
    Trains only adapter layers -> higher efficiency & accuracy on small data.
    """

    # 1) Build input texts
    inputs = []
    for qa in qa_pairs:
        prompt = build_rag_prompt(qa.question, qa.supporting_passages)
        inputs.append(prompt + qa.answer)

    # 2) Tokenize
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(inputs, truncation=True, padding=True, return_tensors="pt")
    dataset = SeqDataset(enc, tokenizer.pad_token_id)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3) Choose LoRA target modules (depends on model type)
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2", "c_proj"]

    # 4) LoRA Config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # 5) Wrap with LoRA adapter
    generator = get_peft_model(generator, lora_config)
    generator.print_trainable_parameters()

    generator.to(device)
    generator.train()

    # 6) Optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, generator.parameters()), lr=lr)

    # 7) Train LoRA layers only
    for epoch in range(epochs):
        epoch_losses = []
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = generator(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        print(f"SFT-LoRA epoch {epoch+1}/{epochs}, loss={np.mean(epoch_losses):.4f}")

    generator.eval()

    # 8) Save adapter
    if isinstance(generator, PeftModel):
        generator.save_pretrained(save_adapter_path)
        print(f"✅ LoRA adapter saved to: {save_adapter_path}")
    else:
        torch.save(generator.state_dict(), f"{save_adapter_path}/generator_full.pt")
        print(f"⚠️ Saved full model state_dict (no PEFT wrapper)")

    return generator
