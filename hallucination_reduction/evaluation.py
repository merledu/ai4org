# hallucination_reduction/evaluation.py

from collections import Counter
from .discriminator import discriminator_predict_text
from .generator import generate_answer
from .retriever import build_rag_prompt
from sklearn.metrics import accuracy_score


def exact_match(a, b):
    """Compute exact match."""
    return int(a.strip().lower() == b.strip().lower())


def f1_score(pred, gold):
    """Compute F1 score between prediction and gold answer."""
    p_tokens = pred.split()
    g_tokens = gold.split()
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens) if p_tokens else 0.0
    recall = num_same / len(g_tokens) if g_tokens else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def overlap_fact_check(answer, supporting_passages):
    """Minimal overlap-based fact check."""
    def overlap_ratio(a, b):
        atoks = set(a.lower().split())
        btoks = set(b.lower().split())
        if not btoks:
            return 0.0
        return len(atoks & btoks) / len(btoks)

    scores = [overlap_ratio(answer, p) for p in supporting_passages]
    return float(sum(scores) / len(scores)) if scores else 0.0


def evaluate_old_vs_new_generator(
    old_gen, new_gen, tokenizer, retriever, qa_pairs, fact_disc, fact_tok, device="cpu"
):
    """
    Evaluate old and new generator models on QA pairs using:
    - Exact match
    - F1
    - Minimal hallucination check
    """
    rows = []
    old_metrics = {"total": 0, "exact": 0, "f1_sum": 0.0, "hallucinated": 0}
    new_metrics = {"total": 0, "exact": 0, "f1_sum": 0.0, "hallucinated": 0}

    for qa in qa_pairs:
        # Retrieve (doc, score) tuples
        retrieved = retriever.retrieve(qa.question, k=3)

        # Generate answers
        old_out = generate_answer(old_gen, tokenizer, retrieved, qa.question, device=device)
        new_out = generate_answer(new_gen, tokenizer, retrieved, qa.question, device=device)

        for label, out, metrics in [("old", old_out, old_metrics), ("new", new_out, new_metrics)]:
            metrics["total"] += 1
            metrics["exact"] += exact_match(out, qa.answer)
            metrics["f1_sum"] += f1_score(out, qa.answer)

            # Discriminator-based fact-check
            fact_pred = discriminator_predict_text(fact_disc, fact_tok, [out], device=device)[0]
            p_fact = fact_pred["probs"][1] if len(fact_pred["probs"]) > 1 else fact_pred["probs"][0]

            # Overlap with supporting passages
            overlap = overlap_fact_check(out, qa.supporting_passages)

            # Minimal hallucination check
            if (p_fact < 0.5) and (overlap < 0.3):
                metrics["hallucinated"] += 1

        rows.append({"question": qa.question, "gold": qa.answer, "old": old_out, "new": new_out})

    old_summary = {
        "exact_match_rate": old_metrics["exact"] / old_metrics["total"],
        "avg_f1": old_metrics["f1_sum"] / old_metrics["total"],
        "hallucination_rate": old_metrics["hallucinated"] / old_metrics["total"]
    }
    new_summary = {
        "exact_match_rate": new_metrics["exact"] / new_metrics["total"],
        "avg_f1": new_metrics["f1_sum"] / new_metrics["total"],
        "hallucination_rate": new_metrics["hallucinated"] / new_metrics["total"]
    }

    return rows, old_summary, new_summary
