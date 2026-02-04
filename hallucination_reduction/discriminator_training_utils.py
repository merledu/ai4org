import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader

from .config import DEVICE, SEED
from .discriminator import SimpleTextDataset, discriminator_predict_text


def train_discriminator_minibatch(
    classifier,
    tokenizer,
    texts,
    labels,
    device=DEVICE,
    epochs=5,
    batch_size=8,
    lr=2e-5,
    val_split=0.2,
):

    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=val_split,
        random_state=SEED,
        stratify=labels if len(set(labels)) > 1 else None,
    )
    ds = SimpleTextDataset(tr_texts, tr_labels, tokenizer)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(classifier.parameters(), lr=lr)
    classifier.train()
    for epoch in range(epochs):
        losses = []
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = classifier(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            loss = nn.CrossEntropyLoss()(outputs.logits, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        classifier.eval()
        val_preds = []
        val_true = []
        for vt, vl in zip(val_texts, val_labels):
            res = discriminator_predict_text(
                classifier, tokenizer, [vt], device=device
            )[0]
            prob_pos = res["probs"][1] if len(res["probs"]) > 1 else res["probs"][0]
            pred = 1 if prob_pos > 0.5 else 0
            val_preds.append(pred)
            val_true.append(int(vl))
        acc = accuracy_score(val_true, val_preds) if len(val_true) > 0 else 0.0
        print(
            f"Disc train epoch {epoch+1}/{epochs}, loss={np.mean(losses):.4f}, val_acc={acc:.4f}"
        )
        classifier.train()
    classifier.eval()
    return classifier


def evaluate_classifier(cls, tokenizer, texts, labels, device=DEVICE):
    if not texts or not labels:
        return {
            "acc": 0.0,
            "prec": 0.0,
            "rec": 0.0,
            "f1": 0.0,
        }
    cls.eval()
    preds = []
    for t in texts:
        res = discriminator_predict_text(cls, tokenizer, [t], device=device)[0]
        p = 1 if res["probs"][1] > 0.5 else 0
        preds.append(p)
    acc = accuracy_score(labels, preds) if len(labels) > 0 else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}
