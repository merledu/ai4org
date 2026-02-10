import json
import os
from dataclasses import dataclass
from typing import List


@dataclass
class QAPair:
    question: str
    answer: str
    supporting_passages: List[str]


def build_corpus_and_qa():
    with open("data/processed/corpus.txt", "r", encoding="utf-8") as f:
        passages = [line.strip() for line in f if line.strip()]

    qa_path = "data/qa/qa.json"
    raw_qas: List[dict] = []
    if os.path.exists(qa_path):
        try:
            with open(qa_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    raw_qas = data
        except Exception:
            raw_qas = []

    base_qa: List[QAPair] = []
    for qa in raw_qas:
        question = qa.get("question")
        answer = qa.get("answer")
        supp = qa.get("supporting_passages") or []
        if isinstance(supp, str):
            supp = [supp]
        if question and answer:
            base_qa.append(QAPair(question, answer, supp))

    if not base_qa:
        for p in passages[:200]:  # keep it bounded
            q = "What does this passage say?"
            a = p[:500]
            base_qa.append(QAPair(q, a, [p]))

    augmented: List[QAPair] = []
    for qa in base_qa:
        augmented.append(qa)
        q1 = qa.question.replace("Can a customer", "Is it possible for a customer")
        q2 = qa.question + " (please advise)"
        a1 = qa.answer
        augmented.append(QAPair(q1, a1, qa.supporting_passages))
        augmented.append(QAPair(q2, a1, qa.supporting_passages))

    return passages, augmented
