import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Device setup
# -------------------------
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        DEVICE = "cuda"
        print(f"✅ {n_gpus} GPUs detected. Using all via device_map='auto'.")
    else:
        DEVICE = "cuda"
        print("✅ Single GPU detected. Using cuda:0")
else:
    DEVICE = "cpu"
    print("⚠️ No GPU detected. Using CPU.")

# -------------------------
# Model & tokenizer loading
# -------------------------
GEN_MODEL = "gpt2"
SAVE_PATH = "./saved_models_improved/generator_final.pt"
CORPUS_PATH = "./data/processed/corpus.txt"

print(f"Loading model/tokenizer: {GEN_MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, device_map="auto")

# Safely load fine-tuned weights
if os.path.exists(SAVE_PATH):
    print(f"✅ Loading fine-tuned weights from {SAVE_PATH}")
    state_dict = torch.load(SAVE_PATH, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"⚠️ Mismatch in keys — missing: {len(missing)}, unexpected: {len(unexpected)}")
else:
    print("⚠️ No fine-tuned weights found, using base GPT-2 model.")

print(f"Model ready. Representative model device: {next(model.parameters()).device}")

# -------------------------
# Corpus loading
# -------------------------
if not os.path.exists(CORPUS_PATH):
    raise FileNotFoundError(f"Corpus file not found: {CORPUS_PATH}")

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    passages = [p.strip() for p in f.readlines() if p.strip()]

print(f"✅ Loaded {len(passages)} passages from corpus.")

# -------------------------
# TF-IDF for retrieval
# -------------------------
vectorizer = TfidfVectorizer(stop_words="english")
passage_vecs = vectorizer.fit_transform(passages)

def retrieve_context(query, top_k=3):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, passage_vecs).flatten()
    top_ids = sims.argsort()[-top_k:][::-1]
    return [(passages[i], float(sims[i])) for i in top_ids]

# -------------------------
# Answer generation
# -------------------------
def generate_answer(question, context):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in decoded:
        decoded = decoded.split("Answer:")[-1].strip()
    return decoded

# -------------------------
# Interactive RAG
# -------------------------
print("\nInteractive enhanced multi-GPU-aware RAG 🧩")
print("Type a question (or 'exit' to quit):\n")

while True:
    question = input("> ").strip()
    if question.lower() in ["exit", "quit"]:
        break

    retrieved = retrieve_context(question)
    combined_ctx = "\n".join([f"[{i+1}] (sim={sim:.2f}) {p}" for i, (p, sim) in enumerate(retrieved)])

    print("\n--- 📚 Context Used ---")
    for i, (p, sim) in enumerate(retrieved):
        print(f"[{i+1}] (sim={sim:.2f}) {p}")

    print("\n--- 🧠 Answer ---")
    answer = generate_answer(question, combined_ctx)
    print(answer)

    print("\n" + "-"*60 + "\n")
