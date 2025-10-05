import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import optim
from torch.utils.data import DataLoader
from typing import List
import numpy as np

from .config import MAX_GEN_TOKENS, MIN_GEN_TOKENS, SFT_EPOCHS, SFT_BATCH, SFT_LR,GEN_MODEL

DEVICE="cuda"
def load_generator(model_name=GEN_MODEL, device=DEVICE):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto")
    model.eval()
    return tokenizer, model

def build_rag_prompt(question: str, retrieved_docs: List[str]) -> str:
    prompt = "### Context:\n"
    for i, doc in enumerate(retrieved_docs, 1):
        prompt += f"[{i}] {doc}\n"
    prompt += "\n### Question:\n" + question + "\n### Answer:\n"
    return prompt

def generate_answer(generator, tokenizer, prompt: str, max_new_tokens=MAX_GEN_TOKENS, min_new_tokens=MIN_GEN_TOKENS,
                    device=DEVICE, num_return_sequences=1, temperature=0.8):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    # Ensure min_new_tokens to avoid empty output collapse
    out = generator.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        do_sample=True,
        top_k=50,
        temperature=temperature,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id
    )
    texts = []
    for seq in out:
        texts.append(tokenizer.decode(seq[inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip())
    return texts


def sft_finetune_generator(generator, tokenizer, qa_pairs, device=DEVICE, epochs=SFT_EPOCHS, batch_size=SFT_BATCH, lr=SFT_LR):
    # Create input sequences: prompt + gold answer
    inputs = []
    for qa in qa_pairs:
        prompt = build_rag_prompt(qa.question, qa.supporting_passages)
        inputs.append(prompt + qa.answer)
    enc = tokenizer(inputs, truncation=True, padding=True, return_tensors="pt")
    dataset = torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"])
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(generator.parameters(), lr=lr)
    generator.train()
    for epoch in range(epochs):
        epoch_losses = []
        for ids, masks in dl:
            ids = ids.to(device); masks = masks.to(device)
            outputs = generator(input_ids=ids, attention_mask=masks, labels=ids)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        print(f"SFT epoch {epoch+1}/{epochs}, loss={np.mean(epoch_losses):.4f}")
    generator.eval()
    return generator
