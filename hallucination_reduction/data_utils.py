import random
import json
from dataclasses import dataclass
from typing import List

@dataclass
class QAPair:
    question: str
    answer: str
    supporting_passages: List[str]

def build_dummy_corpus_and_qa():
    passages = [
        "Our cancellation policy: Customers may cancel within 24 hours for a full refund.",
        "Data retention rules: We keep logs for 90 days and anonymize them after that.",
        "Access control: Only managers can approve employee role changes.",
        "Security policy: All devices must have disk encryption enabled.",
        "Tone guideline: Use professional, concise, and empathetic language in responses.",
    ]
    base_qa = [
        QAPair("Can a customer get a refund if they cancel after 12 hours?",
               "Yes — customers may cancel within 24 hours for a full refund.",
               [passages[0]]),
        QAPair("How long do we retain logs?",
               "We retain logs for 90 days and then anonymize them.",
               [passages[1]]),
        QAPair("Who can approve role changes for employees?",
               "Only managers can approve employee role changes.",
               [passages[2]]),
        QAPair("Do we require disk encryption on devices?",
               "Yes, all devices must have disk encryption enabled.",
               [passages[3]]),
        QAPair("What tone should customer-facing replies use?",
               "Use professional, concise, and empathetic language.",
               [passages[4]]),
    ]
    augmented = []
    for qa in base_qa:
        augmented.append(qa)
        q1 = qa.question.replace("Can a customer", "Is it possible for a customer")
        q2 = qa.question + " (please advise)"
        a1 = qa.answer
        augmented.append(QAPair(q1, a1, qa.supporting_passages))
        augmented.append(QAPair(q2, a1, qa.supporting_passages))
    return passages, augmented

def load_dataset(path: str):
    """
    Load a real QA dataset from a JSON file at `path`.
    JSON format:
    [
      {
        "question": "...",
        "answer": "...",
        "supporting_passages": ["...", "..."]
      },
      ...
    ]
    """
    with open(path, "r") as f:
        data = json.load(f)

    passages = []
    qa_pairs = []
    for item in data:
        passages.extend(item.get("supporting_passages", []))
        qa_pairs.append(
            QAPair(
                question=item["question"],
                answer=item["answer"],
                supporting_passages=item.get("supporting_passages", []),
            )
        )

    # Deduplicate passages
    passages = list(set(passages))
    return passages, qa_pairs
