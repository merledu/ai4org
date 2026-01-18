# Data Generation Pipeline

The `data_generation_pipeline` is a standalone tool designed to transform raw policy documents into high-quality, validated Question-Answer pairs. These pairs serve as the training data for the Hallucination Reduction module.

## 🌟 Features

*   **PDF/TXT Support**: Extracts text from standard policy document formats.
*   **Smart Chunking**: Splits documents into manageable chunks with overlap to ensure no context is lost at boundaries.
*   **LLM-based Generation**: Uses **Qwen 7B** (quantized) to generate intelligent questions and answers.
*   **Evidence Extraction**: Identifies the exact sentences in the source text that support the answer.
*   **Strict Validation**: Automatically rejects Q&A pairs that contain hallucinations (e.g., referencing non-existent section numbers).
*   **Deduplication**: Removes duplicate or highly similar questions using semantic similarity.

## 🛠️ Setup & Installation

The pipeline runs as a separate module.

1.  Navigate to the directory:
    ```bash
    cd data_generation_pipeline/data-pipeline
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 📖 Usage

### Command Line Interface (CLI)

The primary way to use the pipeline is via the CLI.

```bash
python -m cli run --input <path_to_input_file> --output <path_to_output_json>
```

**Example:**
```bash
python -m cli run \
    --input data/input/security_policy.pdf \
    --output data/output/qa_dataset.json
```

### Input Data
Place your raw documents in `data_generation_pipeline/data-pipeline/data/input/`.

### Output Format
The output is a JSON file containing a list of Q&A objects:

```json
[
  {
    "question": "What is the maximum data retention period?",
    "answer": "The maximum data retention period is 7 years.",
    "context": "...data shall be retained for 7 years...",
    "evidence": ["data shall be retained for 7 years"],
    "meta": {
      "source": "security_policy.pdf",
      "chunk_id": 12
    }
  }
]
```

## ⚙️ Configuration

Configuration files are located in `config/`.

### `pipeline_config.yaml`
Controls the processing logic.

```yaml
chunking:
  chunk_size: 512       # Tokens per chunk
  overlap: 128          # Overlap between chunks

generation:
  max_qa_per_chunk: 5   # Max questions to generate per chunk
  temperature: 0.7      # Creativity of the model

deduplication:
  similarity_threshold: 0.85  # Threshold for considering questions duplicates
```

### `model_config.yaml`
Controls the LLM settings.

```yaml
model_name: "Qwen/Qwen-7B-Chat"
quantization: "4bit"    # Use 4-bit quantization for memory efficiency
device: "cuda"          # Use GPU
```

## 🔍 Validation Logic

The pipeline implements several validators in `src/validators.py`:

1.  **Section Checker**: If the answer mentions "Section X.Y", it verifies that "Section X.Y" actually exists in the text chunk.
2.  **Acronym Checker**: Ensures any acronyms used in the answer are defined or present in the context.
3.  **Relevance Checker**: Uses a lightweight cross-encoder to verify the question is actually answered by the text.
