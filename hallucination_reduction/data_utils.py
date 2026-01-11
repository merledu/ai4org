import json
import os
from dataclasses import dataclass
from typing import List


@dataclass
class QAPair:
    """Represents a question-answer pair with supporting evidence.

    This dataclass encapsulates a single QA pair along with the passages
    that support or provide context for the answer. Used throughout the
    hallucination reduction pipeline for training and evaluation.

    Attributes:
        question (str): The question text.
        answer (str): The answer text corresponding to the question.
        supporting_passages (List[str]): List of text passages that provide
            evidence or context for the answer.

    Example:
        >>> qa = QAPair(
        ...     question="What is the refund policy?",
        ...     answer="Refunds are processed within 30 days.",
        ...     supporting_passages=["All refunds must be requested within 30 days..."]
        ... )
        >>> print(qa.question)
        What is the refund policy?
    """

    question: str
    answer: str
    supporting_passages: List[str]


def build_corpus_and_qa():
    """Build corpus and augmented QA pairs from data files.

    This function loads text passages from the corpus file and QA pairs from
    the qa.json file, then performs data augmentation by creating variations
    of existing questions. If no QA pairs exist, it generates basic pairs
    from the first 200 passages.

    The augmentation strategy includes:
    - Original QA pairs
    - Rephrased questions (e.g., "Can a customer" → "Is it possible for a customer")
    - Questions with additional context (e.g., adding "(please advise)")

    Returns:
        Tuple[List[str], List[QAPair]]: A tuple containing:
            - List of all text passages from the corpus
            - List of augmented QA pairs with variations

    Raises:
        FileNotFoundError: If data/processed/corpus.txt doesn't exist.

    Example:
        >>> passages, qa_pairs = build_corpus_and_qa()
        >>> print(f"Loaded {len(passages)} passages")
        Loaded 1523 passages
        >>> print(f"Generated {len(qa_pairs)} QA pairs")
        Generated 450 QA pairs
        >>> print(qa_pairs[0].question)
        What is the refund policy?

    Note:
        - The function performs 3x augmentation (original + 2 variations)
        - If qa.json is missing or corrupted, fallback data is generated
        - Supporting passages from qa.json are preserved in augmented pairs
    """
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
