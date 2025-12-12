from sentence_transformers import SentenceTransformer, util

_embed_model = None

def ensure_embed_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(model_name)
    return _embed_model

def semantic_dedupe(qas: list, threshold: float=0.88):
    if not qas:
        return []
    model = ensure_embed_model()
    questions = [entry["question"] for entry in qas]
    q_embs = model.encode(questions, convert_to_tensor=True)
    keep = []
    used = [False]*len(questions)
    for i in range(len(questions)):
        if used[i]:
            continue
        keep.append(qas[i])
        sims = util.cos_sim(q_embs[i], q_embs).cpu().numpy()[0]
        for j in range(i+1, len(questions)):
            if sims[j] >= threshold:
                used[j] = True
    return keep

