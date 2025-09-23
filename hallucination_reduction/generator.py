import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Absolute imports from your package
from .retriever import build_rag_prompt
from .data_utils import QAPair
from .config import DEVICE, GEN_MODEL, MAX_GEN_TOKENS, MIN_GEN_TOKENS

# ====================================================
# Load Generator
# ====================================================
def load_generator(model_name=GEN_MODEL, device=DEVICE):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


# ====================================================
# Generate Answer
# ====================================================


# hallucination_reduction/generator.py

def generate_answer(generator, tokenizer, retrieved_docs, question, device):
    """
    Generate answer with minimal fact-checking.
    `retrieved_docs` is a list of (doc, score) tuples.
    """
    # Extract only the document text
    docs_only = [doc for doc, _ in retrieved_docs]

    # Build RAG prompt
    prompt = build_rag_prompt(question, retrieved_docs)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    outputs = generator.generate(
        **inputs,
        max_new_tokens=MAX_GEN_TOKENS,
        min_new_tokens=MIN_GEN_TOKENS,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    # Minimal fact check: return exact passage if fully contained
    answer_lower = answer.lower()
    for doc in docs_only:
        if doc.lower() in answer_lower:
            return doc

    # Fallback: pick the retrieved passage with highest word overlap
    overlap_scores = [
        len(set(answer_lower.split()) & set(doc.lower().split())) / max(1, len(doc.split()))
        for doc in docs_only
    ]
    best_idx = overlap_scores.index(max(overlap_scores))
    if max(overlap_scores) > 0.2:  # minimal overlap threshold
        return docs_only[best_idx]

    # Otherwise, return the generated text
    return answer



# ====================================================
# Supervised Fine-Tuning (SFT)
# ====================================================
def train_generator_minibatch(generator, tokenizer, qa_pairs,
                              epochs=3, batch_size=2, lr=3e-5, device="cpu"):
    generator.train()
    optimizer = torch.optim.AdamW(generator.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(0, len(qa_pairs), batch_size):
            batch = qa_pairs[i:i+batch_size]
            texts = [f"Question: {qa.question}\nAnswer: {qa.answer}" for qa in batch]

            # Tokenize once for both input and labels
            encodings = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(device)

            # For GPT-style causal LM, just pass labels=input_ids
            outputs = generator(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(qa_pairs) // batch_size)
        print(f"SFT Epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}")

    generator.eval()
    return generator
