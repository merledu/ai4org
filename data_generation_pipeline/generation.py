import re
import time
from typing import List, Tuple, Dict
from chunking import sentences_from_text
from config import (
    MAX_NEW_TOKENS,
    DETERMINISTIC_TEMP,
    SAMPLING_TEMP,
    EMBED_MODEL,
    EVIDENCE_SENT_TOP_K,
    SEMANTIC_DEDUPE_THRESHOLD,
)


def generate_with_retry(tokenizer, model, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    import torch
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(model.device)
    # deterministic first
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=DETERMINISTIC_TEMP,
                top_p=0.95,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    except Exception:
        text = ""

    # if empty or clearly invalid, retry sampling
    if not text or len(text) < 10:
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=SAMPLING_TEMP,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        except Exception:
            text = ""

    return text

# -------------------------
# Parse Q/A blocks
# -------------------------
Q_RE = re.compile(r'^\s*Q\s*(\d+)\s*[:.\-]\s*(.+?)(\?)?\s*$', re.IGNORECASE)
A_RE = re.compile(r'^\s*A\s*(\d+)\s*[:.\-]\s*(.+)', re.IGNORECASE)

def parse_qa_block(block: str) -> List[Tuple[str,str]]:
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    pairs = []
    i = 0
    while i < len(lines)-1:
        q_match = Q_RE.match(lines[i])
        a_match = A_RE.match(lines[i+1])
        if q_match and a_match and q_match.group(1) == a_match.group(1):
            q = q_match.group(2).strip()
            a = a_match.group(2).strip().rstrip(".")  # normalize
            pairs.append((q, a))
            i += 2
        else:
            i += 1
    return pairs

# -------------------------
# Policy name / section detection (stronger)
# -------------------------
POLICY_NUM_RE = re.compile(r'\b(?:section|sec|clause|article|policy)\s*\d+(?:\.\d+)*\b', re.IGNORECASE)
# Named policy pattern: capitalized words ending with 'Policy' or typical nouns
NAMED_POLICY_RE = re.compile(r'\b([A-Z][A-Za-z&\-]{2,}(?:\s+[A-Z][A-Za-z&\-]{2,}){0,6}\s+(?:Policy|policy))\b')
ABBR_POLICY_RE = re.compile(r'\b(KYC|CDD|EDD|AML|CFT|FD|ATM|STR|CTR)\b', re.IGNORECASE)

def question_has_valid_reference(q: str) -> bool:
    """Return True if question mentions numeric section or a named policy / common policy acronyms."""
    if POLICY_NUM_RE.search(q):
        return True
    if NAMED_POLICY_RE.search(q):
        return True
    if ABBR_POLICY_RE.search(q):
        return True
    return False

def is_vague_question(q: str) -> bool:
    ql = q.lower()
    vague_phrases = ["purpose of this policy", "what is this policy", "what is the purpose of this policy", "what does this policy say"]
    return any(p in ql for p in vague_phrases)

def valid_question(q: str) -> bool:
    if len(q) < 5:
        return False
    if is_vague_question(q) and not question_has_valid_reference(q):
        return False
    return question_has_valid_reference(q)

# -------------------------
# Evidence extraction: choose top sentence(s) from chunk that support the answer
# -------------------------
from sentence_transformers import SentenceTransformer, util
embed_model = None

def ensure_embed_model():
    global embed_model
    if embed_model is None:
        embed_model = SentenceTransformer(EMBED_MODEL)

def extract_evidence_sentences(answer: str, chunk: str, k: int=EVIDENCE_SENT_TOP_K) -> List[str]:
    """
    Return top-k sentences from chunk most similar to answer.
    """
    sents = sentences_from_text(chunk)
    if not sents:
        return []
    ensure_embed_model()
    # embed answer and sentences
    embeddings = embed_model.encode([answer] + sents, convert_to_tensor=True)
    ans_emb = embeddings[0]
    sent_embs = embeddings[1:]
    scores = util.cos_sim(ans_emb, sent_embs)[0].cpu().numpy()
    # pick top-k
    idxs = scores.argsort()[::-1][:k]
    chosen = [sents[i] for i in idxs if scores[i] > 0.1]  # small floor
    return chosen

# -------------------------
# Semantic dedupe for questions (embeddings + threshold)
# -------------------------
def semantic_dedupe(qas: List[Dict], threshold: float=SEMANTIC_DEDUPE_THRESHOLD) -> List[Dict]:
    """
    Remove semantically similar questions. Keeps the first occurrence in case of duplicates.
    """
    if not qas:
        return []
    ensure_embed_model()
    questions = [entry["question"] for entry in qas]
    q_embs = embed_model.encode(questions, convert_to_tensor=True)
    keep = []
    used = [False]*len(questions)
    for i in range(len(questions)):
        if used[i]:
            continue
        keep.append(qas[i])
        sims = util.cos_sim(q_embs[i], q_embs).cpu().numpy()[0]
        # mark all semantically close ones
        for j in range(i+1, len(questions)):
            if sims[j] >= threshold:
                used[j] = True
    return keep

