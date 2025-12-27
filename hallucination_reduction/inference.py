import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys, time
from threading import Thread


CORPUS_PATH = "data/processed/corpus.txt"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
WEIGHTS_DIR = "./saved_models_improved"


def load_corpus(corpus_path=CORPUS_PATH):
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(docs)} documents from corpus.")
    return docs


def build_embeddings(docs, embed_model_name=EMBED_MODEL, device="cpu"):
    embedder = SentenceTransformer(embed_model_name, device=device)
    print("Encoding corpus embeddings...")
    corpus_embeddings = embedder.encode(docs, convert_to_numpy=True, show_progress_bar=True)
    return embedder, corpus_embeddings


def retrieve_relevant_chunks(query, embedder, corpus_embeddings, docs, top_k=3):
    query_emb = embedder.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(query_emb, corpus_embeddings)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    top_docs = [docs[i] for i in top_indices]
    return top_docs


def find_best_model(weights_dir=WEIGHTS_DIR):
    if not os.path.exists(weights_dir):
        print(f"Weights directory not found: {weights_dir}")
        return None

    files = os.listdir(weights_dir)
    candidates = [f for f in files if f.startswith("generator") and f.endswith(".pt")]
    if not candidates:
        print("No generator checkpoints found.")
        return None


    if "generator_final.pt" in candidates:
        return os.path.join(weights_dir, "generator_final.pt")
    if any("best" in f for f in candidates):
        return os.path.join(weights_dir, sorted([f for f in candidates if "best" in f])[0])

    epoch_files = [f for f in candidates if "epoch" in f]
    if epoch_files:
        latest = sorted(epoch_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
        return os.path.join(weights_dir, latest)

    return os.path.join(weights_dir, candidates[0])


def load_model(model_name=BASE_MODEL, weights_dir=WEIGHTS_DIR):
    print(f"Loading base model/tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    weights_path = find_best_model(weights_dir)
    if weights_path and os.path.exists(weights_path):
        print(f"Attempting to load fine-tuned weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")

        
        model_state = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}

        missing = len(model_state) - len(filtered_dict)
        if missing > 0:
            print(f"Skipped {missing} parameters due to shape mismatch.")
        model_state.update(filtered_dict)
        model.load_state_dict(model_state, strict=False)
    else:
        print("No fine-tuned weights found. Using base model only.")

    model.eval()
    print("Model ready. Type: CAUSAL CHAT, Device:", next(model.parameters()).device)
    return model, tokenizer, next(model.parameters()).device


import sys, time
from threading import Thread
from transformers import TextIteratorStreamer

def generate_answer(model, tokenizer, device, question, retrieved_docs, max_new_tokens=1024):
    context = "\n".join(retrieved_docs)
    prompt = f"""
### Context:
{context}

### Question:
{question}

### Instructions:
Provide a long, thoughtful, and well-structured answer based only on the given context. 
Include reasoning, relevant details, and explanations, but avoid speculation or hallucination.

### Answer:
"""

    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print("\n Generator ", end="", flush=True)

    
    generation_thread = Thread(
        target=model.generate,
        kwargs=dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,           
            top_p=0.9,                 
            do_sample=True,
            repetition_penalty=1.05,   
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        ),
    )
    generation_thread.start()

    generated_tokens = []
    for token in streamer:
        generated_tokens.append(token)
        print(token, end="", flush=True)
        time.sleep(0.015)

    generation_thread.join()
    print("\n")
    return "".join(generated_tokens)


def main():
    model, tokenizer, device = load_model()
    docs = load_corpus()
    embedder, corpus_embeddings = build_embeddings(docs, device=device)
    

    print("\n RAG-Enhanced Chat Mode Started — type 'exit' to quit.\n")

    while True:
        question = input(" You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print(" Exiting chat. Goodbye!")
            break

        retrieved_docs = retrieve_relevant_chunks(question, embedder, corpus_embeddings, docs)
        print(f"\n Retrieved {len(retrieved_docs)} relevant context chunks.")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"   [{i}] {doc[:100]}...")

        answer = generate_answer(model, tokenizer, device, question, retrieved_docs)

        print(f"\n generator: {answer}\n")


if __name__ == "__main__":
    main()
