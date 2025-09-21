import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from typing import List

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISC_MODEL = "distilbert-base-uncased"

# -------------------------
# Discriminator loader
# -------------------------
def load_discriminator(model_name=DISC_MODEL, device=DEVICE, num_labels=2):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    ).to(device)
    model.eval()
    return tokenizer, model

# -------------------------
# Dataset wrapper
# -------------------------
class SimpleTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# -------------------------
# Prediction helper
# -------------------------
def discriminator_predict_text(classifier, tokenizer, texts: List[str], device=DEVICE, batch_size=8):
    classifier.eval()
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        # Move all tensors to the device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = classifier(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()  # bring probs back to CPU
            preds = np.argmax(probs, axis=1)
        for p, prob in zip(preds, probs):
            results.append({"pred": int(p), "probs": prob.tolist()})
    return results

# -------------------------
# Train discriminator
# -------------------------
def train_discriminator_minibatch(classifier, tokenizer, texts, labels,
                                 device=DEVICE, epochs=5, batch_size=8, lr=2e-5, val_split=0.2):
    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        texts, labels, test_size=val_split, random_state=42,
        stratify=labels if len(set(labels)) > 1 else None
    )
    ds = SimpleTextDataset(tr_texts, tr_labels, tokenizer)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    classifier.train()
    for epoch in range(epochs):
        losses = []
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = classifier(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = criterion(outputs.logits, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Validation
        classifier.eval()
        val_preds, val_true = [], []
        for vt, vl in zip(val_texts, val_labels):
            res = discriminator_predict_text(classifier, tokenizer, [vt], device=device)[0]
            pred = int(np.argmax(res["probs"]))
            val_preds.append(pred)
            val_true.append(int(vl))
        acc = accuracy_score(val_true, val_preds) if len(val_true) > 0 else 0.0
        print(f"Disc train epoch {epoch+1}/{epochs}, loss={np.mean(losses):.4f}, val_acc={acc:.4f}")
        classifier.train()
    classifier.eval()
    return classifier

# -------------------------
# Evaluate classifier
# -------------------------
def evaluate_classifier(cls, tokenizer, texts, labels, device=DEVICE):
    cls.eval()
    preds = []
    for t in texts:
        res = discriminator_predict_text(cls, tokenizer, [t], device=device)[0]
        p = int(np.argmax(res["probs"]))
        preds.append(p)
    acc = accuracy_score(labels, preds) if len(labels) > 0 else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}
