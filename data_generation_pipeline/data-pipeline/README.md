# data-pipeline
```
A modular, production-ready Question & Answer (Q/A) generation pipeline designed for bank policy documents.
The project transforms large policy PDFs into validated Q/A pairs using LLMs (Qwen 7B), evidence extraction, and semantic deduplication.

---
```

## 🚀 Features

```
- PDF/TXT text extraction
- Advanced cleaning for policy documents
- Configurable chunking with overlap
- Qwen 7B (4-bit) model support via `bitsandbytes`
- Deterministic + sampling generation retries
- Strict question validation (section numbers, policy names, acronyms)
- Q/A parsing with numbering pattern matching
- Sentence-level evidence extraction using `sentence-transformers`
- Exact + semantic deduplication (FAISS cosine similarity)
- Fully modular source code
- CLI interface for terminal execution

---
```

## 📁 Project Structure

```
data-pipeline/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── config/
│   ├── pipeline_config.yaml
│   └── model_config.yaml
├── data/
│   ├── input/
│   │   └── (place input PDF/TXT here)
│   ├── interim/
│   └── output/
├── src/
│   ├── __init__.py
│   ├── cli.py
│   ├── pipeline_runner.py
│   ├── file_loader.py
│   ├── cleaner.py
│   ├── chunker.py
│   ├── prompts.py
│   ├── model_loader.py
│   ├── generator.py
│   ├── qa_parser.py
│   ├── validators.py
│   ├── evidence.py
│   └── dedupe.py
├── notebooks/
│   └── experiments.ipynb
├── tests/
│   ├── test_cleaner.py
│   ├── test_chunker.py
│   └── test_parser.py
└── docs/
    └── architecture.md


````

---
```
## 🛠 Installation

### 1. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# or
.venv\Scripts\activate          # Windows
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚨 Input File

Place your input file (PDF or TXT) inside:

```
data/input/
```

Example:

```
data/input/amanah_bank_policy.pdf
```

---

## ▶ Running the Pipeline

Run via CLI:

```bash
python -m cli run \
    --input data/input/amanah_bank_policy.pdf \
    --output data/output/results.json
```

This command will:

1. Extract document text
2. Clean and normalize it
3. Chunk it into overlapping windows
4. Generate Q/A pairs per chunk
5. Extract supporting evidence sentences
6. Deduplicate similar questions
7. Save results as JSON

---

## ⚙ Configuration

Edit these files to customize the pipeline:

```
config/pipeline_config.yaml   # chunk size, Q/A limits, dedupe threshold
config/model_config.yaml      # model name, embedding model, quantization options
```

---

## 🧪 Running Tests

```bash
pytest -q
```

Tests cover:

* Cleaner
* Chunker
* Q/A parser

---

## 🧠 Notebooks

Use the `notebooks/` directory for:

* debugging
* exploring text chunks
* visualizing embeddings
* evaluating model output

---

## 🤝 Contributing

Pull requests and improvements are welcome.
Follow standard Git branching with PR review.

---

## 📜 License

Open-source — free to use and modify.
