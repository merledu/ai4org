from collections import Counter
from typing import List

import numpy as np

from .config import DEVICE, MAX_GEN_TOKENS, MIN_GEN_TOKENS, TOP_K
from .discriminator import discriminator_predict_text
from .generator import build_rag_prompt, generate_answer


def exact_match(a: str, b: str) -> int:
    return int(a.strip().lower() == b.strip().lower())


def accuracy_score(y_true, y_pred):
    """
    Compute the fraction of correct predictions.
    """
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0):
    """
    Compute precision, recall, F1 score for binary classification.
    Returns (precision, recall, f1, support)
    """
    if not y_true:
        return 0.0, 0.0, 0.0, 0

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    support = sum(1 for t in y_true if t == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
    recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else zero_division
    )

    return precision, recall, f1, support


def f1_score(pred: str, gold: str) -> float:
    p_tokens = pred.split()
    g_tokens = gold.split()
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens) if p_tokens else 0.0
    recall = num_same / len(g_tokens) if g_tokens else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def overlap_fact_check(answer: str, supporting_passages: List[str]) -> float:
    """
    Compute fraction of supporting passage tokens that overlap with the generated answer.
    Returns value in [0,1] average across passages.
    """

    def overlap_ratio(a, b):
        atoks = set(a.lower().split())
        btoks = set(b.lower().split())
        if len(btoks) == 0:
            return 0.0
        return len(atoks & btoks) / len(btoks)

    scores = [overlap_ratio(answer, p) for p in supporting_passages]
    return float(np.mean(scores)) if scores else 0.0


def evaluate_old_vs_new_generator(
    old_gen, new_gen, tokenizer, retriever, qa_pairs, fact_disc, fact_tok, device=DEVICE
):
    rows = []
    old_metrics = {"total": 0, "exact": 0, "f1_sum": 0.0, "hallucinated": 0}
    new_metrics = {"total": 0, "exact": 0, "f1_sum": 0.0, "hallucinated": 0}
    for qa in qa_pairs:
        retrieved = [p for idx, p in retriever.retrieve(qa.question, k=TOP_K)]
        prompt = build_rag_prompt(qa.question, retrieved)
        old_out = generate_answer(
            old_gen,
            tokenizer,
            prompt,
            max_new_tokens=MAX_GEN_TOKENS,
            min_new_tokens=MIN_GEN_TOKENS,
            device=device,
            num_return_sequences=1,
        )[0]
        new_out = generate_answer(
            new_gen,
            tokenizer,
            prompt,
            max_new_tokens=MAX_GEN_TOKENS,
            min_new_tokens=MIN_GEN_TOKENS,
            device=device,
            num_return_sequences=1,
        )[0]

        for _label, out, metrics in [
            ("old", old_out, old_metrics),
            ("new", new_out, new_metrics),
        ]:
            metrics["total"] += 1
            metrics["exact"] += exact_match(out, qa.answer)
            metrics["f1_sum"] += f1_score(out, qa.answer)

            fact_pred = discriminator_predict_text(
                fact_disc, fact_tok, [out], device=device
            )[0]
            p_fact = (
                fact_pred["probs"][1]
                if len(fact_pred["probs"]) > 1
                else fact_pred["probs"][0]
            )
            overlap = overlap_fact_check(out, qa.supporting_passages)

            if (p_fact < 0.5) and (overlap < 0.3):
                metrics["hallucinated"] += 1

        rows.append(
            {"question": qa.question, "gold": qa.answer, "old": old_out, "new": new_out}
        )

    old_summary = {
        "exact_match_rate": old_metrics["exact"] / old_metrics["total"],
        "avg_f1": old_metrics["f1_sum"] / old_metrics["total"],
        "hallucination_rate": old_metrics["hallucinated"] / old_metrics["total"],
    }
    new_summary = {
        "exact_match_rate": new_metrics["exact"] / new_metrics["total"],
        "avg_f1": new_metrics["f1_sum"] / new_metrics["total"],
        "hallucination_rate": new_metrics["hallucinated"] / new_metrics["total"],
    }
    return rows, old_summary, new_summary


def evaluate_classifier(cls, tokenizer, texts, labels, device=DEVICE):
    cls.eval()
    preds = []
    for t in texts:
        res = discriminator_predict_text(cls, tokenizer, [t], device=device)[0]
        p = 1 if res["probs"][1] > 0.5 else 0
        preds.append(p)
    acc = accuracy_score(labels, preds) if len(labels) > 0 else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )  # noqa: F821
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}
