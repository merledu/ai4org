# System Architecture

This document provides a detailed overview of the AI4Org system architecture, including its components, data flow, and design decisions.

## High-Level Overview

AI4Org is designed to reduce hallucinations in Large Language Models (LLMs) used for organizational policy Q&A. It employs a Retrieval-Augmented Generation (RAG) approach enhanced by a multi-discriminator system and Reinforcement Learning (RL).

### Core Components

1.  **Data Generation Pipeline**: Automates the creation of high-quality Q&A pairs from raw policy documents.
2.  **Hallucination Reduction Module**: The core ML engine that trains the generator using SFT and RL.
3.  **Frontend Application**: A cross-platform desktop interface for users to interact with the system.

## Architecture Diagram

```mermaid
graph TB
    subgraph "Data Ingestion & Processing"
        Raw[Raw Documents<br/>(PDF/TXT)] --> Cleaner[Document Cleaner]
        Cleaner --> Chunker[Smart Chunker]
        Chunker --> QGen[Qwen 7B<br/>Q&A Generator]
        QGen --> Evidence[Evidence Extractor]
        Evidence --> Dedupe[Semantic Deduplication]
        Dedupe --> QA_DB[(Validated Q&A Database)]
    end

    subgraph "Training Pipeline"
        QA_DB --> SFT[Supervised Fine-Tuning]
        SFT --> RL_Loop[RL Loop (REINFORCE)]

        subgraph "Discriminator System"
            Fact[Factuality Discriminator]
            Style[Style Discriminator]
            Safety[Safety Discriminator]
        end

        RL_Loop <--> Fact
        RL_Loop <--> Style
        RL_Loop <--> Safety

        RL_Loop --> FinalModel[Final Generator Model]
    end

    subgraph "Inference System"
        User[User Query] --> Retriever[Semantic Retriever]
        Retriever --> Context[Context Retrieval]
        Context --> FinalModel
        FinalModel --> Response[Generated Response]
    end

    style Raw fill:#f9f,stroke:#333,stroke-width:2px
    style QA_DB fill:#bbf,stroke:#333,stroke-width:2px
    style FinalModel fill:#bfb,stroke:#333,stroke-width:2px
```

## Component Details

### 1. Data Generation Pipeline (`data_generation_pipeline`)

The data pipeline is critical for creating the "ground truth" needed for training.

*   **Input**: Raw PDF or TXT files (e.g., bank policies).
*   **Processing**:
    *   **Cleaning**: Removes headers, footers, and page numbers.
    *   **Chunking**: Splits text into overlapping windows (default 512 tokens) to preserve context.
    *   **Generation**: Uses a quantized Qwen 7B model to generate question-answer pairs from each chunk.
    *   **Validation**: Checks for hallucinated section numbers or acronyms not present in the text.
*   **Output**: A JSON file containing Q&A pairs linked to their source evidence.

### 2. Hallucination Reduction Module (`hallucination_reduction`)

This module implements the novel training approach.

*   **Generator**: Based on `TinyLlama-1.1B-Chat`. It is small enough for efficient training but capable enough for coherent text generation.
*   **Discriminators**: Three separate `DistilBERT` models trained to classify:
    *   **Factuality**: Is the answer supported by the provided context?
    *   **Style**: Is the tone professional and appropriate?
    *   **Safety**: Is the content safe and non-toxic?
*   **Reinforcement Learning**: Uses the REINFORCE algorithm. The generator produces an answer, the discriminators score it, and these scores are used as rewards to update the generator's policy.

### 3. Frontend (`frontend`)

The user interface for the system.

*   **Technology**: `pywebview` allows building a desktop app using web technologies (HTML/CSS/JS) while running a Python backend.
*   **Features**:
    *   **Chat Interface**: Similar to standard LLM chat UIs.
    *   **Admin Dashboard**: For monitoring usage and managing users.
    *   **Local Inference**: Runs the Python inference engine directly on the user's machine.

## Design Decisions

### Why TinyLlama?
We chose TinyLlama-1.1B because it offers an excellent balance between performance and resource requirements. It can be fine-tuned on consumer hardware (like a single RTX 3090 or even lower with quantization), making the system accessible for smaller organizations.

### Why Separate Discriminators?
Using a single reward model can lead to "reward hacking" where the model optimizes for the score rather than the actual quality. By separating factuality, style, and safety, we can tune the weights of each reward component independently, ensuring a more balanced output.

### Why Qwen 7B for Data Generation?
Qwen 7B has shown superior performance in following complex instructions compared to other models in its size class. Its ability to generate structured Q&A pairs from unstructured text is key to our pipeline's success.
