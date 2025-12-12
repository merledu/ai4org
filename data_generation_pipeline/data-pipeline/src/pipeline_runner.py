import json
import time
from pathlib import Path
from loguru import logger

from .file_loader import extract_text
from .cleaner import clean_text
from .chunker import chunk_text
from .prompts import build_prompt
from .model_loader import load_model_tokenizer
from .generator import generate_with_retry
from .qa_parser import parse_qa_block
from .validators import valid_question
from .evidence import extract_evidence_sentences
from .dedupe import semantic_dedupe

def run_pipeline(input_path: str, out_file: str, cfg: dict):
    logger.info("Loading text from {}", input_path)
    raw = extract_text(input_path)
    logger.info("Raw text length (chars): {}", len(raw))
    cleaned = clean_text(raw)
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    with open("data/interim/cleaned_bank_corpus.txt", "w", encoding="utf-8") as f:
        f.write(cleaned)
    logger.info("Saved cleaned_bank_corpus.txt")

    chunks = chunk_text(cleaned, cfg.get("chunk_size_words", 120), cfg.get("chunk_overlap_words", 30))
    logger.info("Total chunks: {} (chunk_size {}, overlap {})", len(chunks), cfg.get("chunk_size_words"), cfg.get("chunk_overlap_words"))

    model_cfg = {}
    try:
        # model_config is optional in cfg
        model_cfg = cfg.get("model_config", {})
    except Exception:
        model_cfg = {}

    # Load model & tokenizer
    logger.info("Loading model and tokenizer — this may take a minute...")
    try:
        tokenizer, model = load_model_tokenizer(cfg.get("default_model", "Qwen/Qwen2.5-7B-Instruct"), model_cfg.get("quantization", {}))
    except Exception as e:
        logger.error("Model load failed: {}", e)
        raise

    results = []
    seen_exact = set()

    for idx, chunk in enumerate(chunks):
        prompt = build_prompt(chunk, max_q=cfg.get("max_q_per_chunk", 5))
        text = generate_with_retry(
            tokenizer, model, prompt,
            max_new_tokens=cfg.get("max_new_tokens", 512),
            deterministic_temp=cfg.get("deterministic_temp", 0.0),
            sampling_temp=cfg.get("sampling_temp", 0.4)
        )
        logger.info("Processed chunk {}/{} — got {} chars", idx+1, len(chunks), len(text))

        if not text or len(text) < 5:
            time.sleep(cfg.get("sleep_between_chunks", 0.12))
            continue

        parsed_pairs = parse_qa_block(text)
        if not parsed_pairs:
            time.sleep(cfg.get("sleep_between_chunks", 0.12))
            continue

        for q, a in parsed_pairs:
            q_norm = q.strip().rstrip("?").strip()
            a_norm = a.strip()
            if not valid_question(q_norm):
                continue
            key = (q_norm.lower(), a_norm.lower())
            if key in seen_exact:
                continue
            seen_exact.add(key)
            evidences = extract_evidence_sentences(a_norm, chunk, k=cfg.get("evidence_sent_top_k", 2))
            results.append({
                "question": q_norm,
                "answer": a_norm,
                "supporting_passages": evidences if evidences else [chunk]
            })
        time.sleep(cfg.get("sleep_between_chunks", 0.12))

    logger.info("Raw generated Q/A count before semantic dedupe: {}", len(results))
    results = semantic_dedupe(results, threshold=cfg.get("semantic_dedupe_threshold", 0.88))
    logger.info("Q/A count after semantic dedupe: {}", len(results))

    Path("data/output").mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Saved final dataset -> {}", out_file)
    return results

