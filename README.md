# Hallucination Reduction (RAG + Discriminators + RL)

This repository implements a retrieval-augmented generation pipeline with discriminators and REINFORCE to reduce hallucinations. It includes:

- Retriever (TF-IDF)
- Generator (HuggingFace causal LM, e.g. gpt2)
- Three discriminators (fact, style, safety) using sequence classification models
- Supervised fine-tuning (SFT) + REINFORCE loop
- Overlap-based fact-check reward augmentation
- Modular structure for real datasets

## Quick start

1. Create virtualenv:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
