import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

CORPUS_PATH = "data/processed/corpus.txt"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # lightweight and accurate

# -----------------------------
# 1. Load or encode corpus
# -----------------------------
def load_corpus(corpus_path=CORPUS_PATH):
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"❌ Corpus file not found: {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]
    print(f"📚 Loaded {len(docs)} documents from corpus.")
    return docs


def build_embeddings(docs, embed_model_name=EMBED_MODEL, device="cpu"):
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(embed_model_name, device=device)
    print("🔹 Encoding corpus embeddings...")
    corpus_embeddings = embedder.encode(docs, convert_to_numpy=True, show_progress_bar=True)
    return embedder, corpus_embeddings


def retrieve_relevant_chunks(query, embedder, corpus_embeddings, docs, top_k=3):
    query_emb = embedder.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(query_emb, corpus_embeddings)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    top_docs = [docs[i] for i in top_indices]
    return top_docs


# -----------------------------
# 2. Load fine-tuned model
# -----------------------------
def find_best_model(weights_dir="./saved_models_improved"):
    if not os.path.exists(weights_dir):
        print(f"⚠️ Weights directory not found: {weights_dir}")
        return None

    files = os.listdir(weights_dir)
    candidates = [f for f in files if f.startswith("generator") and f.endswith(".pt")]
    if not candidates:
        print("⚠️ No generator checkpoints found.")
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


def load_model(model_name="gpt2", weights_dir="./saved_models_improved"):
    print(f"🔹 Loading base model/tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    weights_path = find_best_model(weights_dir)
    if weights_path and os.path.exists(weights_path):
        print(f"✅ Loading fine-tuned weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    else:
        print("⚠️ No fine-tuned weights found. Using base model only.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"✅ Model ready. Type: CAUSAL, Device: {device}")
    return model, tokenizer, device


# -----------------------------
# 3. Generate with retrieved context
# -----------------------------
def generate_answer(model, tokenizer, device, question, retrieved_docs, max_length=256):
    context = "\n".join(retrieved_docs)
    prompt = f"Context:\n{context}\n\nQ: {question}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("A:")[-1].strip()


# -----------------------------
# 4. Interactive RAG chat
# -----------------------------
def main():
    model, tokenizer, device = load_model()
    docs = load_corpus()
    embedder, corpus_embeddings = build_embeddings(docs, device=device)

    print("\n💬 RAG-Enhanced Chat Mode Started — type 'exit' to quit.\n")

    while True:
        question = input("❓ You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("👋 Exiting chat. Goodbye!")
            break

        retrieved_docs = retrieve_relevant_chunks(question, embedder, corpus_embeddings, docs)
        print(f"\n🔎 Retrieved {len(retrieved_docs)} relevant context chunks.")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"   [{i}] {doc[:100]}...")

        answer = generate_answer(model, tokenizer, device, question, retrieved_docs)
        print(f"\n🤖 Bot: {answer}\n")


if __name__ == "__main__":
    main()
