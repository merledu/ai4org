import re 
import time
from pathlib import Path
from typing import List, Dict, Tuple
from file_loader import extract_text, get_file_name_from_dir
from cleaner import clean_text, clean_text_leakage
from chunker import chunk_text
from prompts import build_prompt
from model_loader import load_model_tokenizer
from generator import generate_with_retry
from qa_parser import parse_qa_block
from validators import valid_question
from evidence import extract_evidence_sentences
from config_reader import load_config


cfg = load_config("config/pipeline_config.yaml")
SLEEP_BETWEEN_CHUNKS = cfg.get("sleep_between_chunks", 0.12)


VAGUE_PATTERNS = [
    r"\bthis policy\b",
    r"\bthis context\b",
    r"\bthis document\b",
    r"\bthis section\b",
    r"\bthe above\b",
    r"\bhere\b"
]


def process_document(file_path: Path, tokenizer, model) -> Tuple[List[Dict], Dict]:
  doc_meta = {
    "doc_id": f"doc_{hash(str(file_path)) & 0xffffffff}",
    "file_name": file_path.name,
    "file_path": str(file_path),
    "file_type": file_path.suffix.lower().lstrip(".")
  }

  try:
    print(f"[INFO] Loading text of {file_path}...")
    raw = extract_text(str(file_path))
    cleaned = clean_text(raw)

    file_name = get_file_name_from_dir(str(file_path))

    # Save cleaned corpus for audit
    with open(f"cleaned_corpus_{file_name}.txt", "w", encoding="utf-8") as f:
        f.write(cleaned)
    print(f"[INFO] Saved cleaned_corpus_{file_name}.txt")


    chunks = chunk_text(cleaned)

    results = []
    seen_exact = set()

    for idx, chunk in enumerate(chunks[:30]):
        prompt = build_prompt(chunk)
        text = generate_with_retry(tokenizer, model, prompt)
        print(f"Working on chunk {idx} out of {len(chunks)} of file {file_name}")
        if not text:
            print(f"[WARN] Empty model output for chunk {idx}")
            continue
        parsed = parse_qa_block(text)
        if not parsed:
            # no Q/A found — skip
            time.sleep(SLEEP_BETWEEN_CHUNKS)
            continue

        for q, a in parsed:
            q = clean_text_leakage(q)
            a = clean_text_leakage(a)

            if any(re.search(p, q.lower()) for p in VAGUE_PATTERNS):
                continue

            if not q or not a:
                continue
            if q.startswith("<") or a.startswith("<"):
                continue

            if len(a.split()) > 60:
                continue


            q_norm = q.strip().rstrip("?")
            a_norm = a.strip()

            if not valid_question(q_norm):
                continue

            key = (q_norm.lower(), a_norm.lower())
            if key in seen_exact:
                continue
            seen_exact.add(key)

            evidences = extract_evidence_sentences(a_norm, chunk)

            results.append({
                **doc_meta,
                "chunk_id": idx,
                "question": q_norm,
                "answer": a_norm,
                "supporting_passages": evidences or [chunk]
            })

    return results, {"status": "success", "qas": len(results)}

  except Exception as e:
      return [], {"status": "failed", "error": str(e)}
