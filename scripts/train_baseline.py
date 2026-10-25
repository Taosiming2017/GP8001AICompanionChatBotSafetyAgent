"""TF-IDF + logistic regression baseline (CPU, seconds). Usually run through scripts/train.py.

    python scripts/train_baseline.py                      # data/synthetic -> runs/full/models/baseline

Serves two purposes: a baseline row in the results table, and a fallback
classifier for the demo if the transformer is not available on the demo laptop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common  # noqa: F401
import joblib
import numpy as np
from _common import DATA, FULL, read_jsonl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from safety.schema import LABELS
from safety.text import normalize_for_model


def make_pipe(C: float) -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, lowercase=True)),
    ])
    return Pipeline([("feats", feats), ("lr", LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(DATA / "train.jsonl"))
    ap.add_argument("--val", default=str(DATA / "val.jsonl"))
    ap.add_argument("--out", default=str(FULL / "models/baseline"))
    args = ap.parse_args()

    tr, va = read_jsonl(args.train), read_jsonl(args.val)
    Xtr = [normalize_for_model(r["text"]) for r in tr]
    Xva = [normalize_for_model(r["text"]) for r in va]
    ytr = np.array([r["level"] for r in tr])
    yva = np.array([r["level"] for r in va])

    best = None
    for C in (0.5, 1, 2, 4, 8, 16):
        pipe = make_pipe(C).fit(Xtr, ytr)
        pred = pipe.predict(Xva)
        f1 = f1_score(yva, pred, average="macro")
        rec3 = recall_score(yva == 3, pred == 3) if (yva == 3).any() else 0.0
        score = f1 + rec3  # model selection: macro-F1 plus crisis recall
        print(f"C={C:<4} macro-F1={f1:.3f} crisis-recall={rec3:.3f}")
        if best is None or score > best[0]:
            best = (score, C, pipe, f1, rec3)

    _, C, pipe, f1, rec3 = best
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "model.joblib")
    np.save(out / "val_logits.npy", np.log(np.clip(pipe.predict_proba(Xva), 1e-9, 1)))
    np.save(out / "val_labels.npy", yva)
    (out / "meta.json").write_text(json.dumps({
        "model_type": "tfidf_lr", "labels": LABELS, "C": C, "val_macro_f1": f1, "val_crisis_recall": rec3,
        "n_train": len(tr), "n_val": len(va)}, indent=2), encoding="utf-8")
    print(f"saved {out}  (C={C}, val macro-F1={f1:.3f}, crisis recall={rec3:.3f})")
    print("next: python scripts/calibrate.py --model-dir", out)


if __name__ == "__main__":
    main()
