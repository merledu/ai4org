import json
from .data_structures import QAPair

def save_corpus(passages, path="data/processed/corpus.txt"):
    with open(path, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(p + "\n\n")

def save_qa(qa_pairs, path="data/qa/qa.json"):
    qa_dicts = [qa.__dict__ for qa in qa_pairs]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(qa_dicts, f, indent=2)
