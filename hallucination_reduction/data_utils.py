from dataclasses import dataclass
from typing import List, Tuple
import random


# -------------------------
# Data structures & dummy private corpus (augment a bit)
# -------------------------
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
