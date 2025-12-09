# AI4Org: Hallucination Reduction & Data Pipeline

This project implements a comprehensive system for reducing hallucinations in Large Language Models (LLMs) using Retrieval-Augmented Generation (RAG), Discriminator-Guided Reinforcement Learning, and a robust data cleaning pipeline. It also includes a desktop frontend for interacting with the system.

## Project Structure

The project is organized into the following main components:

*   **`hallucination_reduction/`**: The core machine learning pipeline. It includes:
    *   **RAG**: Retrieval-Augmented Generation using TF-IDF/Embeddings.
    *   **Discriminators**: Classifiers for Factuality, Style, and Safety.
    *   **RL Loop**: A REINFORCE-based loop to fine-tune the generator based on discriminator feedback.
    *   [Read more](./hallucination_reduction/README.md)

*   **`frontend/`**: A desktop application built with Python (`pywebview`) and HTML/CSS/JS to interact with the model.
    *   [Read more](./frontend/README.md)

*   **`data_cleaning_pipeline/`**: Scripts and tools for processing and cleaning the raw data used for training.

*   **`tests/`**: Unit and integration tests for the project.

## Prerequisites

*   Python 3.10 or higher
*   CUDA-capable GPU (recommended for training)

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd ai4org
    ```

2.  Create a virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

### Training the Model

To train the hallucination reduction pipeline (Discriminators -> SFT -> RL):

```bash
python -m hallucination_reduction.main
```

### Running Inference

To chat with the trained model via the command line:

```bash
python -m hallucination_reduction.inference
```

### Running the Frontend

To launch the desktop application:

```bash
cd frontend
pip install -r requirements.txt
python main.py
```

## Contributing

Please read [CONTRIBUTING.md](./CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.
