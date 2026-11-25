"""Modèle 2 : fine-tuning de BERT pour la classification titre -> tag."""
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from src.data import MODELS_DIR, SEED, load_dataset, split

MAX_LEN = 48  # les titres dépassent rarement ~30 tokens


def encode(tokenizer, texts, labels=None):
    enc = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=MAX_LEN, return_tensors="pt")
    tensors = [enc["input_ids"], enc["attention_mask"]]
    if labels is not None:
        tensors.append(torch.tensor(labels))
    return TensorDataset(*tensors)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for input_ids, mask, *_ in loader:
        logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits
        preds.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="bert-base-uncased",
                   help="ex. distilbert-base-uncased pour aller ~2x plus vite sur CPU")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    args = p.parse_args()

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_dataset()
    X_train, X_test, y_train, y_test = split(df)
    labels = sorted(set(y_train))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(labels), id2label=id2label, label2id=label2id,
    ).to(device)

    train_loader = DataLoader(encode(tokenizer, X_train, [label2id[y] for y in y_train]),
                              batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(encode(tokenizer, X_test), batch_size=64)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total), total)

    print(f"Device={device} | train={len(X_train)} test={len(X_test)} | {len(labels)} tags", flush=True)
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for input_ids, mask, y in train_loader:
            out = model(input_ids=input_ids.to(device), attention_mask=mask.to(device),
                        labels=y.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += out.loss.item()
        y_pred = [id2label[i] for i in predict(model, test_loader, device)]
        print(f"epoch {epoch + 1}/{args.epochs} loss={running / len(train_loader):.4f} "
              f"acc={accuracy_score(y_test, y_pred):.4f} "
              f"f1_macro={f1_score(y_test, y_pred, average='macro'):.4f}", flush=True)

    print(classification_report(y_test, y_pred, zero_division=0))
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "model_name": args.model_name,
        "epochs": args.epochs,
    }
    print(metrics)

    out_dir = MODELS_DIR / "bert"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (MODELS_DIR / "bert_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Modèle sauvegardé dans", out_dir)


if __name__ == "__main__":
    main()
