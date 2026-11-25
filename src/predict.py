"""Prédire le tag d'un titre avec l'un ou l'autre modèle.

    python -m src.predict "How to parse JSON in Python?"
    python -m src.predict --model bert "How to parse JSON in Python?"
    python -m src.predict            # mode interactif
"""
import argparse
from functools import lru_cache

import joblib
import numpy as np

from src.data import MODELS_DIR


@lru_cache
def _baseline():
    return joblib.load(MODELS_DIR / "baseline.joblib")


@lru_cache
def _bert():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = MODELS_DIR / "bert"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    return tok, model


def predict_tags(title: str, model: str = "baseline", top_k: int = 3):
    """Renvoie [(tag, probabilité), ...] du plus probable au moins probable."""
    if model == "baseline":
        clf = _baseline()
        probs = clf.predict_proba([title])[0]
        classes = list(clf.classes_)
    elif model == "bert":
        import torch
        tok, m = _bert()
        with torch.no_grad():
            enc = tok([title], return_tensors="pt", truncation=True, max_length=48)
            probs = torch.softmax(m(**enc).logits, -1)[0].numpy()
        classes = [m.config.id2label[i] for i in range(len(probs))]
    else:
        raise ValueError("model doit être 'baseline' ou 'bert'")
    idx = np.argsort(probs)[::-1][:top_k]
    return [(classes[i], float(probs[i])) for i in idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("title", nargs="?", help="titre à classer (sinon mode interactif)")
    p.add_argument("--model", choices=["baseline", "bert"], default="baseline")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    if args.title:
        titles = [args.title]
    else:
        titles = iter(lambda: input("Titre > ").strip(), "")
    for t in titles:
        preds = predict_tags(t, args.model, args.top_k)
        details = ", ".join(f"{tag} {prob:.0%}" for tag, prob in preds)
        print(f"-> {preds[0][0]}   ({details})")


if __name__ == "__main__":
    main()
