# Architecture

- file_loader: reads PDF/TXT
- cleaner: document cleaning + regex heuristics
- chunker: fixed-size chunking with overlap
- prompts: prompt templates for LLM
- model_loader: loads quantized LLM
- generator: deterministic + sampling retry logic
- qa_parser: extract Q/A pairs from model output
- validators: question checks (policy refs)
- evidence: sentence-level evidence using sentence-transformers
- dedupe: exact + semantic dedupe

The implementation mirrors the earlier single-file pipeline and is modularized for maintainability. :contentReference[oaicite:2]{index=2}

