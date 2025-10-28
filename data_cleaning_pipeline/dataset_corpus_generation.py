# data_cleaning_pipeline/main.py

import os
import json
import time
from tqdm import tqdm
from openai import OpenAI
from pathlib import Path

# =============================
# CONFIG
# =============================
OPENAI_API_KEY = "AIzaSyAuLmVPLvoRYub9d-gLgHjbVzzDp4xXJlQ"
MODEL = "gpt-4o-mini"

DATA_DIR = "data/raw"
QA_OUTPUT_FILE = "data/qa/qa.json"
CORPUS_FILE = "data/processed/corpus.txt"

CHUNK_SIZE = 1500  # characters per chunk

client = OpenAI(api_key=OPENAI_API_KEY)

# =============================
# HELPERS
# =============================

def read_text_from_file(file_path):
    """Read raw text from .txt, .pdf, .docx, etc."""
    text = ""
    if file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    elif file_path.lower().endswith(".pdf"):
        import PyPDF2
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif file_path.lower().endswith(".docx"):
        from docx import Document
        doc = Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs)
    return text.strip()


def chunk_text(text, chunk_size=1500):
    """Split document text into smaller overlapping chunks."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


def safe_openai_call(prompt, max_retries=10):
    """Call OpenAI API with retry on rate limits or network errors."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a precise and exhaustive QA dataset generator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_message = str(e)
            if "rate_limit" in error_message or "429" in error_message:
                wait_time = 60 * (attempt + 1)
                print(f"⚠️ Rate limit hit. Waiting {wait_time}s before retry #{attempt+1}...")
                time.sleep(wait_time)
                continue
            print(f"❌ Unexpected error: {e}")
            time.sleep(10)
    return None


def generate_qas_for_chunk(text_chunk):
    """Generate many QA pairs for one text chunk."""
    prompt = f"""
You are generating a large, high-quality QA dataset.

Given the following document passage:

---
{text_chunk}
---

Generate 10–15 *unique, detailed, and accurate* QA pairs in JSON list format:
[
  {{
    "question": "...",
    "answer": "..."
  }},
  ...
]

Focus on comprehension, reasoning, procedural steps, definitions, comparisons, and hypothetical scenarios.
"""
    result = safe_openai_call(prompt)
    if not result:
        return []
    try:
        data = json.loads(result)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        # try to extract JSON manually if the model added text around it
        try:
            start = result.find("[")
            end = result.rfind("]")
            data = json.loads(result[start:end+1])
            return data
        except Exception:
            return []
    return []


def append_to_json(file_path, data_list):
    """Append a list of JSON objects to a JSON file (maintaining valid JSON array)."""
    Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
        existing.extend(data_list)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)


def append_to_corpus(file_path, chunks):
    Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.strip() + "\n\n")


# =============================
# MAIN PIPELINE
# =============================

def process_document(doc_path):
    name = os.path.basename(doc_path).replace(".pdf", "").replace(".docx", "").replace(".txt", "")
    print(f"\n📘 Processing: {name}")

    text = read_text_from_file(doc_path)
    chunks = chunk_text(text, CHUNK_SIZE)

    append_to_corpus(CORPUS_FILE, chunks)
    print(f"✏️ Added {len(chunks)} chunks to corpus file.")

    all_qas = []
    for chunk in tqdm(chunks, desc=f"Generating QAs for {name}"):
        qas = generate_qas_for_chunk(chunk)
        if qas:
            all_qas.extend(qas)
            append_to_json(QA_OUTPUT_FILE, qas)
    print(f"✅ Saved {len(all_qas)} QAs for {name}")


def main():
    print("🚀 Starting QA + Corpus generation...")
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith((".pdf", ".docx", ".txt"))]

    for f in files:
        process_document(f)

    print("\n🎯 Done! All QAs saved to:")
    print(f"   → {QA_OUTPUT_FILE}")
    print("   And corpus appended to:")
    print(f"   → {CORPUS_FILE}")


if __name__ == "__main__":
    main()
