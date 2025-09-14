import os
from datasets import load_dataset
import evaluate
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler
)
from tqdm.auto import tqdm
from torch.optim import AdamW


def train_discriminator():
    dataset_path = "./data/datasets/qa_dataset.jsonl"

    # 1. Load dataset
    dataset = load_dataset("json", data_files=dataset_path)
    print(dataset)

    # 2. Load tokenizer
    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # 3. Tokenize QA pairs
    def preprocess(example):
        text = f"Q: {example['question']} A: {example['answer']}"
        tokens = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256
        )
        return tokens

    encoded = dataset["train"].map(preprocess)

    # 4. Add labels (binary: does it have supporting text?)
    def add_labels(example):
        example["labels"] = 1 if example["supporting_text"].strip() else 0
        return example

    encoded = encoded.map(add_labels)

    # 5. Set format for PyTorch
    encoded.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    # 6. DataLoader
    train_loader = DataLoader(encoded, batch_size=16, shuffle=True)

    # 7. Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 8. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_loader) * 3  # 3 epochs
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # 9. Metric
    accuracy = evaluate.load("accuracy")

    # 10. Training loop
    progress_bar = tqdm(range(num_training_steps))
    model.train()

    for epoch in range(3):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        print(f"Epoch {epoch+1} completed ✅ Loss: {loss.item():.4f}")

    # 11. Save model
    save_dir = "./models/discriminator"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"✅ Discriminator saved to {save_dir}")


if __name__ == "__main__":
    train_discriminator()
