"""Fine-tune DistilRoBERTa as the 4-level safety classifier. Usually run through scripts/train.py.

    python scripts/train_transformer.py                                  # data/synthetic -> runs/full/models/transformer
    python scripts/train_transformer.py --model prajjwal1/bert-tiny --epochs 2   # quick CPU smoke test

Input  = (user message, "[scenario] previous companion line") as a sentence pair.
Loss   = class-weighted cross-entropy; the crisis class gets an extra weight
         (--crisis-weight) because a missed crisis costs more than a false alarm.
Select = best epoch by (validation macro-F1 + crisis recall).
Output = runs/full/models/transformer/{model files, tokenizer, meta.json, val_logits.npy, val_labels.npy, train_log.json}

Rough cost with ~850 training messages, 4 epochs, max_len 128:
  NVIDIA GPU: 1-3 minutes.   CPU only: roughly 10-20 minutes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import torch
from _common import DATA, FULL, read_jsonl, set_seed
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from safety.classifier import format_context
from safety.schema import LABELS
from safety.text import normalize_for_model


class PairDataset(Dataset):
    def __init__(self, rows):
        self.a = [normalize_for_model(r["text"]) for r in rows]
        self.b = [format_context(r.get("context"), r.get("scenario")) for r in rows]
        self.y = [int(r["level"]) for r in rows]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.a[i], self.b[i], self.y[i]


def make_collate(tok, max_len):
    def collate(batch):
        a, b, y = zip(*batch)
        enc = tok(list(a), list(b), truncation="only_second", max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y)
        return enc
    return collate


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    out, ys = [], []
    for batch in loader:
        ys.append(batch.pop("labels").numpy())
        batch = {k: v.to(device) for k, v in batch.items()}
        out.append(model(**batch).logits.float().cpu().numpy())
    return np.concatenate(out), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilroberta-base")
    ap.add_argument("--train", default=str(DATA / "train.jsonl"))
    ap.add_argument("--val", default=str(DATA / "val.jsonl"))
    ap.add_argument("--out", default=str(FULL / "models/transformer"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--crisis-weight", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    tr, va = read_jsonl(args.train), read_jsonl(args.val)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)},
    ).to(device)

    collate = make_collate(tok, args.max_len)
    dl_tr = DataLoader(PairDataset(tr), batch_size=args.batch, shuffle=True, collate_fn=collate)
    dl_va = DataLoader(PairDataset(va), batch_size=64, shuffle=False, collate_fn=collate)

    # class weights: inverse frequency, normalised to mean 1, crisis multiplied by --crisis-weight
    counts = np.bincount([int(r["level"]) for r in tr], minlength=len(LABELS)).astype(float)
    w = counts.sum() / (len(LABELS) * np.maximum(counts, 1))
    w[3] *= args.crisis_weight
    w = w / w.mean()
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float, device=device))
    print("class counts:", counts.astype(int).tolist(), "weights:", np.round(w, 2).tolist())

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=args.lr)
    steps = len(dl_tr) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * steps), steps)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log, best = [], None
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for batch in dl_tr:
            y = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(**batch).logits.float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(y)

        logits, yv = predict_logits(model, dl_va, device)
        pred = logits.argmax(1)
        f1 = f1_score(yv, pred, average="macro")
        rec3 = recall_score(yv == 3, pred == 3) if (yv == 3).any() else 0.0
        row = {"epoch": ep, "train_loss": total / len(tr), "val_macro_f1": f1, "val_crisis_recall": rec3,
               "seconds": round(time.time() - t0, 1)}
        log.append(row)
        print(json.dumps(row))
        if best is None or f1 + rec3 > best:
            best = f1 + rec3
            model.save_pretrained(out)
            tok.save_pretrained(out)
            np.save(out / "val_logits.npy", logits)
            np.save(out / "val_labels.npy", yv)
            (out / "meta.json").write_text(json.dumps({
                "model_type": "transformer", "base_model": args.model, "labels": LABELS, "max_len": args.max_len,
                "best_epoch": ep, "val_macro_f1": f1, "val_crisis_recall": rec3, "args": vars(args),
                "n_train": len(tr), "n_val": len(va)}, indent=2), encoding="utf-8")

    (out / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"saved best epoch to {out}. next: python scripts/calibrate.py --model-dir {out}")


if __name__ == "__main__":
    main()
