from typing import List, Tuple
from transformers import pipeline as hf_pipeline
from .data_structures import QAPair
from .loader import load_documents
from .preprocessing import split_into_passages

# HuggingFace models
qg_model = "valhalla/t5-base-qg-hl"
qa_model = "deepset/roberta-base-squad2"

qg = hf_pipeline("text2text-generation", model=qg_model)
qa = hf_pipeline("question-answering", model=qa_model)

def build_corpus_and_qa(folder: str, num_qs: int = 2) -> Tuple[List[str], List[QAPair]]:
    """Build corpus + QA dataset from raw documents folder."""
    raw_text = load_documents(folder)
    passages = split_into_passages(raw_text)

    qa_pairs: List[QAPair] = []
    for passage in passages:
        input_text = f"generate questions: {passage}"
        outputs = qg(input_text, max_length=64, num_return_sequences=num_qs)

        for out in outputs:
            q = out["generated_text"].strip()
            try:
                ans = qa(question=q, context=passage)["answer"]
            except Exception:
                ans = "Answer not found"
            qa_pairs.append(QAPair(q, ans, [passage]))

    return passages, qa_pairs
