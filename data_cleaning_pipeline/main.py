# data_cleaning_pipeline/main.py
import os, re, json, fitz, math
from tqdm import tqdm
from transformers import pipeline
from difflib import SequenceMatcher
from collections import Counter

# ---------------- CONFIG ----------------
RAW_DIR = "../data/raw"
PROCESSED_DIR = "../data/processed"
QA_DIR = "../data/qa"

CORPUS_PATH = os.path.join(PROCESSED_DIR, "corpus.txt")
QA_PATH = os.path.join(QA_DIR, "qa_clean.json")

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"  # change to flan-t5-large for 16 GB cards
MAX_CHUNK_LEN = 900
QAS_PER_CHUNK = 6
AUGMENT_PER_QA = 2
MIN_SCORE = 0.55      # minimum clarity/relevance threshold

# ---------------- STEP 1: PDF EXTRACTION ----------------
def extract_text_from_pdfs(raw_dir):
    texts = []
    for file in os.listdir(raw_dir):
        if file.lower().endswith(".pdf"):
            print(f"📄 Extracting from {file}")
            try:
                with fitz.open(os.path.join(raw_dir, file)) as doc:
                    texts.append("\n".join(page.get_text("text") for page in doc))
            except Exception as e:
                print(f"⚠️ {file}: {e}")
    return "\n".join(texts)

# ---------------- STEP 2: CLEANING ----------------
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1. \2", text)
    for b in ["•", "", "●"]:
        text = text.replace(b, "-")
    return text.strip()

# ---------------- STEP 3: CHUNKING ----------------
def chunk_text(text, max_len=MAX_CHUNK_LEN):
    sents = re.split(r"(?<=[.!?]) +", text)
    chunks, buf = [], ""
    for s in sents:
        if len(buf) + len(s) < max_len:
            buf += " " + s
        else:
            chunks.append(buf.strip())
            buf = s
    if buf:
        chunks.append(buf.strip())
    return chunks

# ---------------- STEP 4: PRIMARY QA GENERATION ----------------
def generate_qa(chunks, model_name=MODEL_NAME, qas_per_chunk=QAS_PER_CHUNK):
    gen = pipeline("text2text-generation", model=model_name, device_map="auto")
    out = []
    for chunk in tqdm(chunks, desc="Primary QA"):
        prompt = f"""
From the banking passage below, create {qas_per_chunk} realistic question–answer pairs in JSON array form.
Each should contain:
  "question", "answer", "supporting_passages"
Add relevant topic tags like [aml kyc #99aa11], [customer support #22bb88].

Passage:
{chunk}
"""
        try:
            txt = gen(prompt, max_new_tokens=1024, temperature=0.9, do_sample=True)[0]["generated_text"]
            m = re.search(r"\[.*\]", txt, re.DOTALL)
            if m:
                out.extend(json.loads(m.group(0)))
        except Exception as e:
            print("skip", e)
    return out

# ---------------- STEP 5: AUGMENTATION ----------------
def augment_dataset(data, model_name=MODEL_NAME, n=AUGMENT_PER_QA):
    gen = pipeline("text2text-generation", model=model_name, device_map="auto")
    aug = []
    for d in tqdm(data, desc="Augmenting"):
        q, a = d["question"], d["answer"]
        ctx = d.get("supporting_passages", [""])[0]
        for _ in range(n):
            p = f'Paraphrase this banking QA as JSON (same keys): {{"question":"{q}","answer":"{a}","supporting_passages":["{ctx}"]}}'
            try:
                t = gen(p, max_new_tokens=256, do_sample=True, temperature=1.0)[0]["generated_text"]
                m = re.search(r"\{.*\}", t, re.DOTALL)
                if m:
                    aug.append(json.loads(m.group(0)))
            except Exception:
                continue
    return data + aug

# ---------------- STEP 6: POST-PROCESSING ----------------
BANK_TERMS = {"aml","kyc","account","loan","transaction","customer","policy",
              "support","refund","compliance","security","risk","data","privacy"}

def similarity(a,b): return SequenceMatcher(None,a,b).ratio()

def score_entry(q,a):
    # clarity: length balance, punctuation
    lq, la = len(q.split()), len(a.split())
    clarity = 1 - abs(lq-15)/40 - abs(la-25)/50
    # relevance: keyword overlap
    rel = sum(k in (q+a).lower() for k in BANK_TERMS)/len(BANK_TERMS)
    return max(0,min(1, (clarity+rel)/2))

def deduplicate_and_filter(data, min_score=MIN_SCORE):
    print("🧹 Cleaning & scoring QAs ...")
    seen, clean = [], []
    for d in tqdm(data, desc="Filtering"):
        q,a = d.get("question",""), d.get("answer","")
        s = score_entry(q,a)
        if s < min_score: continue
        if any(similarity(q,x["question"])>0.85 for x in seen): continue
        d["score"]=round(s,3)
        seen.append(d)
        clean.append(d)
    print(f"✅ {len(clean)} clean entries kept of {len(data)} total.")
    return clean

# ---------------- STEP 7: SAVE ----------------
def save_dataset(corpus,data):
    os.makedirs(PROCESSED_DIR,exist_ok=True)
    os.makedirs(QA_DIR,exist_ok=True)
    with open(CORPUS_PATH,"w",encoding="utf-8") as f: f.write(corpus)
    with open(QA_PATH,"w",encoding="utf-8") as f: json.dump(data,f,indent=2,ensure_ascii=False)
    print(f"✅ Saved corpus → {CORPUS_PATH}")
    print(f"✅ Saved filtered dataset → {QA_PATH} ({len(data)} items)")

# ---------------- MAIN ----------------
def main():
    print("🏦 Running full data-generation pipeline ...")
    raw = extract_text_from_pdfs(RAW_DIR)
    corpus = clean_text(raw)
    chunks = chunk_text(corpus)
    base = generate_qa(chunks)
    aug  = augment_dataset(base)
    clean = deduplicate_and_filter(aug)
    save_dataset(corpus, clean)
    print("🎯 Finished — high-quality banking QA dataset ready!")

if __name__ == "__main__":
    main()
