from loaders import extract_text
from chunking import clean_text, chunk_text, sentences_from_text
from prompt import build_prompt
from generation import generate_with_retry, parse_qa_block
from model_utils import load_model_tokenizer
from generation import valid_question, extract_evidence_sentences, semantic_dedupe
import json
import time
from config import (
    FILE_PATH,
    CHUNK_SIZE_WORDS,
    CHUNK_OVERLAP_WORDS,
    DEFAULT_MODEL,
    MAX_Q_PER_CHUNK,
    EVIDENCE_SENT_TOP_K,
    SEMANTIC_DEDUPE_THRESHOLD,
    SLEEP_BETWEEN_CHUNKS,
)


def run_pipeline(input_path: str, out_file: str):
    print("[INFO] Loading text...")
    raw = extract_text(input_path)
    print(f"[INFO] Raw text length (chars): {len(raw)}")
    cleaned = clean_text(raw)
    # Save cleaned corpus for audit
    with open("cleaned_bank_corpus.txt", "w", encoding="utf-8") as f:
        f.write(cleaned)
    print("[INFO] Saved cleaned_bank_corpus.txt")

    # Build sentence map for evidence extraction if needed
    full_sentences = sentences_from_text(cleaned)

    # Chunking
    chunks = chunk_text(cleaned, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
    print(f"[INFO] Total chunks: {len(chunks)} (chunk size {CHUNK_SIZE_WORDS} words, overlap {CHUNK_OVERLAP_WORDS})")

    # Load model & tokenizer
    print("[INFO] Loading model and tokenizer (8-bit, device_map='auto') — this may take a minute...")
    tokenizer, model = load_model_tokenizer(DEFAULT_MODEL)
    print("[INFO] Model loaded.")


    results = []
    seen_exact = set()  # exact text dedupe for speed

    for idx, chunk in enumerate(chunks[:50]):
        prompt = build_prompt(chunk, max_q=MAX_Q_PER_CHUNK)
        text = generate_with_retry(tokenizer, model, prompt)
        print(f"Working on chunk {idx} out of {len(chunks)}")
        if not text:
            print(f"[WARN] Empty model output for chunk {idx}")
            continue
        parsed_pairs = parse_qa_block(text)
        if not parsed_pairs:
            # no Q/A found — skip
            time.sleep(SLEEP_BETWEEN_CHUNKS)
            continue

        for q, a in parsed_pairs:
            # Normalize question text (strip trailing punctuation)
            q_norm = q.strip().rstrip("?").strip()
            a_norm = a.strip()

            # validation
            if not valid_question(q_norm):
                continue

            # exact dedupe
            key = (q_norm.lower(), a_norm.lower())
            if key in seen_exact:
                continue
            seen_exact.add(key)

            # evidence extraction (top sentences)
            evidences = extract_evidence_sentences(a_norm, chunk, k=EVIDENCE_SENT_TOP_K)

            results.append({
                "question": q_norm,
                "answer": a_norm,
                "supporting_passages": evidences if evidences else [chunk]
            })

        time.sleep(SLEEP_BETWEEN_CHUNKS)

    print(f"[INFO] Raw generated Q/A count before semantic dedupe: {len(results)}")

    # Semantic dedupe (embedding-based) to remove near-duplicates
    results = semantic_dedupe(results, threshold=SEMANTIC_DEDUPE_THRESHOLD)
    print(f"[INFO] Q/A count after semantic dedupe: {len(results)}")

    # Final save
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved final dataset -> {out_file}")
    return results



if __name__=="__main__":
    run_pipeline(FILE_PATH, "results.json")
