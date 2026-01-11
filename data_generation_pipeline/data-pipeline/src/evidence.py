from typing import List

import nltk
from sentence_transformers import SentenceTransformer, util

_embed_model = None


def ensure_embed_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(model_name)
    return _embed_model


def sentences_from_text(text: str):
    try:
        nltk.data.find("tokenizers/punkt")
    except Exception:
        nltk.download("punkt")
    from nltk.tokenize import sent_tokenize

    sents = sent_tokenize(text)
    return [s.strip() for s in sents if s.strip()]


def extract_evidence_sentences(
    answer: str,
    chunk: str,
    k: int = 2,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[str]:
    sents = sentences_from_text(chunk)
    if not sents:
        return []
    model = ensure_embed_model(model_name)
    embeddings = model.encode([answer] + sents, convert_to_tensor=True)
    ans_emb = embeddings[0]
    sent_embs = embeddings[1:]
    scores = util.cos_sim(ans_emb, sent_embs)[0].cpu().numpy()
    idxs = scores.argsort()[::-1][:k]
    chosen = [sents[i] for i in idxs if scores[i] > 0.1]
    return chosen
