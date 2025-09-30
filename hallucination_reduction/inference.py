import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# -------------------------
# Dummy corpus
# -------------------------
passages = [
    "Our cancellation policy: Customers may cancel within 24 hours for a full refund.",
    "Data retention rules: We keep logs for 90 days and anonymize them after that.",
    "Access control: Only managers can approve employee role changes.",
    "Security policy: All devices must have disk encryption enabled.",
    "Tone guideline: Use professional, concise, and empathetic language in responses.",
]

# -------------------------
# Simple TF-IDF retriever
# -------------------------
class SimpleRetriever:
    def __init__(self, passages):
        self.passages = passages
        self.vectorizer = TfidfVectorizer().fit(passages)
        self.vectors = self.vectorizer.transform(passages)

    def retrieve(self, query, k=3):
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.vectors)[0]
        idxs = sims.argsort()[-k:][::-1]
        return [(self.passages[i], sims[i]) for i in idxs]

retriever = SimpleRetriever(passages)

# -------------------------
# Load generator
# -------------------------
GEN_MODEL_NAME = "gpt2"
GEN_MODEL_PATH = "./saved_models_improved/generator_final.pt"  # adjust path

tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
generator = AutoModelForCausalLM.from_pretrained(GEN_MODEL_NAME).to(DEVICE)
generator.load_state_dict(torch.load(GEN_MODEL_PATH, map_location=DEVICE))
generator.eval()

# -------------------------
# RAG prompt builder
# -------------------------
def build_rag_prompt(question, retrieved_docs):
    prompt = "### Context:\n"
    for i, (doc, _) in enumerate(retrieved_docs, 1):
        prompt += f"[{i}] {doc}\n"
    prompt += "\n### Question:\n" + question + "\n### Answer:\n"
    return prompt

# -------------------------
# Generate answer
# -------------------------
def generate_answer(prompt, max_new_tokens=64, min_new_tokens=5):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out = generator.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        do_sample=True,
        top_k=50,
        temperature=0.8,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return text

# -------------------------
# Light overlap-based fact check
# -------------------------
def overlap_fact_check(answer, supporting_passages):
    answer_tokens = set(answer.lower().split())
    scores = []
    for p in supporting_passages:
        passage_tokens = set(p.lower().split())
        if len(passage_tokens) > 0:
            scores.append(len(answer_tokens & passage_tokens) / len(passage_tokens))
    return float(sum(scores)/len(scores)) if scores else 0.0

# -------------------------
# Typing effect function
# -------------------------
def type_out(text, delay=0.03):
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()

# -------------------------
# Real-time loop with highlighted context
# -------------------------
def answer_question(question):
    retrieved = retriever.retrieve(question)
    prompt = build_rag_prompt(question, retrieved)
    answer = generate_answer(prompt)

    overlap_score = overlap_fact_check(answer, [doc for doc, _ in retrieved])

    print("\n--- Answer ---")
    type_out(answer)  # simulate typing

    print("\n--- Context Used ---")
    for i, (doc, sim) in enumerate(retrieved, 1):
        print(f"[{i}] ({sim:.2f}) {doc}")

    print("\n--- Minimal Fact Check ---")
    print(f"Context overlap: {overlap_score:.2f}")
    if overlap_score < 0.2:
        print("⚠️ Warning: Answer may be hallucinated (low overlap with context)")

print("Interactive enhanced fast mode: Type a question (or 'exit' to quit):")
while True:
    q = input("\n> ")
    if q.strip().lower() == "exit":
        break
    answer_question(q)
