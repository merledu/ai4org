from dataclasses import dataclass
from typing import List, Tuple
import random
import json

# -------------------------
# Data structures & dummy private corpus (augment a bit)
# -------------------------
@dataclass
class QAPair:
    question: str
    answer: str
    supporting_passages: List[str]

def build_dummy_corpus_and_qa():
    # Load passages from file
    with open("data/processed/corpus.txt", "r", encoding="utf-8") as f:
        passages = [line.strip() for line in f if line.strip()]

    # Load base QA dataset from file
    with open("data/qa/qa.json", "r", encoding="utf-8") as f:
        raw_qas = json.load(f)

    base_qa = [
        QAPair(
            qa["question"],
            qa["answer"],
            qa["supporting_passages"]
        )
        for qa in raw_qas
    ]

    # Small augmentation: produce paraphrases (template-based)
    augmented = []
    for qa in base_qa:
        augmented.append(qa)
        # paraphrase question (simple)
        q1 = qa.question.replace("Can a customer", "Is it possible for a customer")
        q2 = qa.question + " (please advise)"
        a1 = qa.answer
        augmented.append(QAPair(q1, a1, qa.supporting_passages))
        augmented.append(QAPair(q2, a1, qa.supporting_passages))

    return passages, augmented
