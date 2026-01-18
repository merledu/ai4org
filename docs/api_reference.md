# API Reference

This document provides a reference for the key classes and functions in the AI4Org codebase.

## Hallucination Reduction

### `hallucination_reduction.generator.Generator`

Wrapper for the TinyLlama causal language model.

#### `__init__(self, model_name: str, device: str)`
Initializes the generator model.
*   `model_name`: Hugging Face model identifier.
*   `device`: 'cuda' or 'cpu'.

#### `generate(self, prompt: str, max_new_tokens: int = 128) -> str`
Generates text based on the prompt.

### `hallucination_reduction.discriminator.Discriminator`

Wrapper for the DistilBERT binary classifier.

#### `__init__(self, model_name: str, device: str)`
Initializes the discriminator model.

#### `predict(self, text: str) -> float`
Returns the probability (0.0 to 1.0) of the positive class.

### `hallucination_reduction.retriever.Retriever`

Handles semantic search.

#### `retrieve(self, query: str, top_k: int = 3) -> List[str]`
Retrieves the top-k most relevant document chunks for the query.

## Data Pipeline

### `data_pipeline.src.pipeline_runner.PipelineRunner`

Orchestrates the data generation process.

#### `run(self, input_path: str, output_path: str)`
Executes the full pipeline on the input file.

### `data_pipeline.src.cleaner.DocumentCleaner`

#### `clean(self, text: str) -> str`
Removes noise, headers, and footers from the text.

### `data_pipeline.src.chunker.SmartChunker`

#### `chunk(self, text: str) -> List[str]`
Splits text into overlapping chunks while respecting sentence boundaries.
