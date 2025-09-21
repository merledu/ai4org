import torch
import time
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------------------------
# Device
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# -------------------------
# Load dataset
# -------------------------
def load_dataset(path="data/raw/real_qa.json"):
    with open(path, "r") as f:
        return json.load(f)

dataset = load_dataset()
passages = list({p for item in dataset for p in item.get("supporting_passages", [])})
print(f"Loaded {len(passages)} passages from dataset.")

# -------------------------
# Hybrid Retriever: TF-IDF + FAISS
# -------------------------
class HybridRetriever:
    def __init__(self, passages, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.passages = passages
        
        # TF-IDF
        self.tf_vectorizer = TfidfVectorizer().fit(passages)
        self.tf_vectors = self.tf_vectorizer.transform(passages)
        
        # FAISS embeddings
        self.embed_model = SentenceTransformer(embedding_model_name)
        self.faiss_embeddings = self.embed_model.encode(passages, convert_to_numpy=True)
        self.index = faiss.IndexFlatIP(self.faiss_embeddings.shape[1])
        self.index.add(self.faiss_embeddings)

    def retrieve(self, query, k=10, alpha=0.5):
        # TF-IDF similarity
        q_vec = self.tf_vectorizer.transform([query])
        tf_sims = cosine_similarity(q_vec, self.tf_vectors)[0]

        # FAISS similarity
        q_emb = self.embed_model.encode([query], convert_to_numpy=True)
        faiss_sims, faiss_ids = self.index.search(q_emb, k)
        faiss_sims = faiss_sims[0]
        faiss_ids = faiss_ids[0]

        combined_scores = []
        for i, passage in enumerate(self.passages):
            score_faiss = faiss_sims[list(faiss_ids).index(i)] if i in faiss_ids else 0
            score_tf = tf_sims[i]
            combined_scores.append((i, alpha*score_faiss + (1-alpha)*score_tf))

        topk = sorted(combined_scores, key=lambda x: x[1], reverse=True)[:k]
        return [(self.passages[i], score) for i, score in topk]

retriever = HybridRetriever(passages)

# -------------------------
# Load RL-tuned generator
# -------------------------
GEN_MODEL_PATH = "./saved_models_improved/generator_rl"  # path to RL-tuned model folder

tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
generator = AutoModelForCausalLM.from_pretrained(GEN_MODEL_PATH, use_safetensors=True).to(DEVICE)
generator.eval()
print("Loaded RL-tuned generator.")

# -------------------------
# RAG prompt builder
# -------------------------
def build_rag_prompt(question, retrieved_docs):
    prompt = "### Context:\n"
    for i, (doc, _) in enumerate(retrieved_docs, 1):
        prompt += f"[{i}] {doc}\n"
    prompt += f"\n### Question:\n{question}\n### Answer:\n"
    return prompt

# -------------------------
# Generate answer
# -------------------------
def generate_answer(prompt, max_new_tokens=128, min_new_tokens=5):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out = generator.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return text

# -------------------------
# Overlap-based fact check
# -------------------------
def overlap_score(answer, passages):
    answer_tokens = set(answer.lower().split())
    scores = []
    for p in passages:
        tokens = set(p.lower().split())
        if tokens:
            scores.append(len(answer_tokens & tokens) / len(tokens))
    return float(sum(scores)/len(scores)) if scores else 0.0

# -------------------------
# Filter retrieved docs by overlap
# -------------------------
def filter_retrieved_docs(answer, retrieved, threshold=0.2):
    filtered = []
    for doc, score in retrieved:
        overlap = len(set(answer.lower().split()) & set(doc.lower().split())) / max(len(set(doc.lower().split())), 1)
        if overlap >= threshold:
            filtered.append((doc, score))
    return filtered if filtered else retrieved

# -------------------------
# Typing effect
# -------------------------
def type_out(text, delay=0.02):
    for c in text:
        print(c, end="", flush=True)
        time.sleep(delay)
    print()

# -------------------------
# Interactive QA
# -------------------------
print("Interactive QA (FAISS + TF-IDF + RL-tuned generator): Type a question (or 'exit'):")

while True:
    q = input("\n> ")
    if q.strip().lower() == "exit":
        break

    retrieved = retriever.retrieve(q)
    filtered_docs = filter_retrieved_docs(q, retrieved, threshold=0.05)
    
    prompt = build_rag_prompt(q, filtered_docs)
    answer = generate_answer(prompt)
    
    # Post-check
    final_docs = filter_retrieved_docs(answer, filtered_docs, threshold=0.15)
    final_prompt = build_rag_prompt(q, final_docs)
    final_answer = generate_answer(final_prompt)

    overlap = overlap_score(final_answer, [doc for doc, _ in final_docs])

    print("\n--- Answer ---")
    type_out(final_answer)

    print("\n--- Context Used ---")
    for i, (doc, sim) in enumerate(final_docs, 1):
        print(f"[{i}] ({sim:.2f}) {doc}")

    print("\n--- Minimal Fact Check ---")
    print(f"Context overlap: {overlap:.2f}")
    if overlap < 0.2:
        print("⚠️ Warning: Answer may be hallucinated (low overlap with context)")
