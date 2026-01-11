from typing import List

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import (
    DEVICE,
    GEN_MODEL,
    MAX_GEN_TOKENS,
    MIN_GEN_TOKENS,
    SFT_BATCH,
    SFT_EPOCHS,
    SFT_LR,
)


def load_generator(model_name=GEN_MODEL, device=DEVICE):
    print(f"Loading generator: {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if we can use 4-bit quantization
    use_4bit = False
    try:
        pass

        if device == "cuda":
            use_4bit = True
            print("Bitsandbytes available, using 4-bit quantization")
    except ImportError:
        print("Bitsandbytes not found, disabling 4-bit quantization")

    try:
        if use_4bit:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_4bit=True,
                torch_dtype=torch.float16,
            )
        else:
            # Fallback to standard loading
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            model.to(device)

    except Exception as e:
        print(f"Error loading model with default settings: {e}")
        print("Falling back to CPU/FP32...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to("cpu")

    model.eval()
    return tokenizer, model


def build_rag_prompt(question: str, retrieved_docs: List[str]) -> str:
    prompt = "### Context:\n"
    for i, doc in enumerate(retrieved_docs, 1):
        prompt += f"[{i}] {doc}\n"
    prompt += "\n### Question:\n" + question + "\n### Answer:\n"
    return prompt


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
    texts = []
    for seq in out:
        texts.append(
            tokenizer.decode(
                seq[inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
        )
    return texts


def sft_finetune_generator(
    generator,
    tokenizer,
    qa_pairs,
    device=DEVICE,
    epochs=SFT_EPOCHS,
    batch_size=SFT_BATCH,
    lr=SFT_LR,
):

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
            ids = ids.to(device)
            masks = masks.to(device)
            outputs = generator(input_ids=ids, attention_mask=masks, labels=ids)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        print(f"SFT epoch {epoch+1}/{epochs}, loss={np.mean(epoch_losses):.4f}")
    generator.eval()
    return generator
