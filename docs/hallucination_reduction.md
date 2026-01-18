# Hallucination Reduction Module

The `hallucination_reduction` module is the core machine learning component of AI4Org. It implements a sophisticated training pipeline designed to minimize hallucinations in Large Language Models.

## 🧠 Model Architecture

### Generator
*   **Base Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
*   **Type**: Causal Language Model (CLM)
*   **Role**: Generates answers based on the provided context and user question.

### Discriminators
The system uses three distinct binary classifiers based on `distilbert-base-uncased`:

1.  **Factuality Discriminator**:
    *   **Input**: `[CLS] context [SEP] answer [SEP]`
    *   **Task**: Predicts if the answer is factually supported by the context.
    *   **Label 1**: Factual / **Label 0**: Hallucinated

2.  **Style Discriminator**:
    *   **Input**: `[CLS] answer [SEP]`
    *   **Task**: Predicts if the answer follows the desired professional style.
    *   **Label 1**: Professional / **Label 0**: Unprofessional

3.  **Safety Discriminator**:
    *   **Input**: `[CLS] answer [SEP]`
    *   **Task**: Predicts if the answer is safe and harmless.
    *   **Label 1**: Safe / **Label 0**: Unsafe

## 🔄 Training Pipeline

The training process consists of three sequential stages:

### 1. Discriminator Training
Before we can use the discriminators to guide the generator, they must be trained themselves.
*   **Data**: A labeled dataset of (context, answer, label) tuples.
*   **Process**: Standard supervised classification training using Cross-Entropy Loss.

### 2. Supervised Fine-Tuning (SFT)
The generator is first fine-tuned on high-quality Q&A pairs to learn the basic task.
*   **Data**: Validated Q&A pairs from the data generation pipeline.
*   **Loss**: Standard Causal Language Modeling (CLM) loss (next-token prediction).

### 3. Reinforcement Learning (RL)
This is where the hallucination reduction happens.
*   **Algorithm**: REINFORCE (Policy Gradient).
*   **Process**:
    1.  Generator produces an answer for a given prompt.
    2.  Discriminators evaluate the answer and provide scores (0-1).
    3.  **Reward Calculation**: $R = w_f \cdot S_{fact} + w_s \cdot S_{style} + w_{safe} \cdot S_{safe}$
    4.  The generator's weights are updated to maximize this expected reward.

## ⚙️ Configuration

All hyperparameters are defined in `hallucination_reduction/config.py`.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SFT_EPOCHS` | 4 | Number of epochs for Supervised Fine-Tuning |
| `DISC_EPOCHS` | 4 | Number of epochs for Discriminator training |
| `RL_EPOCHS` | 4 | Number of epochs for RL training |
| `MC_ROLLOUTS` | 6 | Monte Carlo rollouts for variance reduction in RL |
| `FACT_WEIGHT` | 0.8 | Weight of the factuality reward (highest priority) |
| `STYLE_WEIGHT` | 0.15 | Weight of the style reward |
| `SAFETY_WEIGHT` | 0.05 | Weight of the safety reward |

## 🚀 Usage

### Training
To run the full training pipeline:
```bash
python -m hallucination_reduction.main
```

### Inference
To chat with the trained model:
```bash
python -m hallucination_reduction.inference
```

## 📂 File Structure

*   `main.py`: Orchestrates the entire training workflow.
*   `generator.py`: Wrapper class for the TinyLlama generator.
*   `discriminator.py`: Wrapper class for the DistilBERT discriminators.
*   `rl_utils.py`: Implementation of the REINFORCE algorithm and reward calculation.
*   `retriever.py`: Handles semantic search and context retrieval.
*   `data_utils.py`: Functions for loading and processing datasets.
