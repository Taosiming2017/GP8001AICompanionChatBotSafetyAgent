"""Layer 2: a small fine-tuned text classifier (4 levels).

Two interchangeable back-ends share one interface:
  * TransformerClassifier - fine-tuned DistilRoBERTa (scripts/train_transformer.py)
  * TfidfClassifier       - TF-IDF + logistic regression baseline (scripts/train_baseline.py)

Both return calibrated probabilities over [safe, tense, overwhelmed, crisis].
ClassifierLayer turns probabilities into a decision with two crisis thresholds:

    p_crisis >= crisis_hi           -> Level 3 straight away
    crisis_lo <= p_crisis < hi      -> grey zone, ask the LLM
    otherwise                       -> argmax level (LLM only if an uncertain Level 2)

Thresholds and the softmax temperature live in <model_dir>/thresholds.json and are
written by scripts/calibrate.py from the validation set.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .schema import LABELS, LayerResult, Level
from .text import normalize_for_model

DEFAULT_THRESHOLDS = {"temperature": 1.0, "crisis_hi": 0.85, "crisis_lo": 0.15, "min_conf": 0.55}


def format_context(context: Optional[str], scenario: Optional[str]) -> str:
    """Second text segment fed to the model: scenario tag + previous companion line."""
    scen = (scenario or "free_chat").strip()
    ctx = normalize_for_model(context or "")
    return f"[{scen}] {ctx}".strip()


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class BaseClassifier:
    name = "base"

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        th = self.model_dir / "thresholds.json"
        self.thresholds = dict(DEFAULT_THRESHOLDS)
        if th.exists():
            self.thresholds.update(json.loads(th.read_text(encoding="utf-8")))

    def logits(self, texts: Sequence[str], contexts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, texts, contexts, temperature: Optional[float] = None) -> np.ndarray:
        T = self.thresholds["temperature"] if temperature is None else temperature
        return softmax(self.logits(texts, contexts) / T)


class TfidfClassifier(BaseClassifier):
    name = "tfidf_lr"

    def __init__(self, model_dir: Path):
        super().__init__(model_dir)
        import joblib

        self.pipe = joblib.load(self.model_dir / "model.joblib")

    def logits(self, texts, contexts) -> np.ndarray:
        # The bag-of-words baseline uses the user message only.
        x = [normalize_for_model(t) for t in texts]
        p = self.pipe.predict_proba(x)
        return np.log(np.clip(p, 1e-9, 1.0))


class TransformerClassifier(BaseClassifier):
    name = "transformer"

    def __init__(self, model_dir: Path, device: Optional[str] = None, max_len: int = 128):
        super().__init__(model_dir)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(self.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        want = device if device and device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = model.to(want).eval()
            self.device = want
        except RuntimeError as e:  # typically CUDA out of memory because LM Studio holds the GPU
            if want == "cpu":
                raise
            print(f"[safety] classifier cannot use {want} ({str(e).splitlines()[0][:80]}); using CPU")
            torch.cuda.empty_cache()
            self.model = model.to("cpu").eval()
            self.device = "cpu"
        meta = self.model_dir / "meta.json"
        self.max_len = json.loads(meta.read_text(encoding="utf-8")).get("max_len", max_len) if meta.exists() else max_len

    def logits(self, texts, contexts, batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            a = [normalize_for_model(t) for t in texts[i:i + batch_size]]
            b = list(contexts[i:i + batch_size])
            enc = self.tok(a, b, truncation="only_second", max_length=self.max_len,
                           padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                out.append(self.model(**enc).logits.float().cpu().numpy())
        return np.concatenate(out, axis=0)


def load_classifier(model_dir: str | Path, device: Optional[str] = None) -> Optional[BaseClassifier]:
    """Load whatever is in model_dir; None if it does not hold a trained model.

    device: None/"auto" (GPU if free, else CPU), "cpu" or "cuda"; transformer only.
    """
    d = Path(model_dir)
    meta = d / "meta.json"
    if not meta.exists():
        return None
    kind = json.loads(meta.read_text(encoding="utf-8")).get("model_type")
    if kind == "tfidf_lr":
        return TfidfClassifier(d)
    if kind == "transformer":
        return TransformerClassifier(d, device=device)
    raise ValueError(f"unknown model_type {kind!r} in {meta}")


class ClassifierLayer:
    def __init__(self, clf: BaseClassifier):
        self.clf = clf
        self.th = clf.thresholds

    def assess(self, text: str, context: Optional[str], scenario: Optional[str]) -> LayerResult:
        t0 = time.perf_counter()
        p = self.clf.predict_proba([text], [format_context(context, scenario)])[0]
        probs = {name: round(float(v), 4) for name, v in zip(LABELS, p)}
        p_crisis = float(p[Level.CRISIS])
        top = int(np.argmax(p))
        conf = float(p[top])
        reasons, wants_llm, risk = [], False, False

        if p_crisis >= self.th["crisis_hi"]:
            level = Level.CRISIS
            reasons.append(f"p_crisis {p_crisis:.2f} >= crisis_hi {self.th['crisis_hi']:.2f}")
        elif p_crisis >= self.th["crisis_lo"] or top == Level.CRISIS:
            level = Level(top)
            wants_llm = risk = True   # a possible crisis: the LLM decides, and fail-safe applies if it cannot
            reasons.append(f"p_crisis {p_crisis:.2f} in grey zone [{self.th['crisis_lo']:.2f}, {self.th['crisis_hi']:.2f})")
        else:
            level = Level(top)
            if level == Level.OVERWHELMED and conf < self.th["min_conf"]:
                wants_llm = True      # unsure between 1 and 2 only; not a crisis signal
                reasons.append(f"uncertain overwhelmed (conf {conf:.2f})")

        return LayerResult(
            layer="classifier", level=level, confidence=conf, probs=probs, reasons=reasons,
            wants_llm=wants_llm, risk_signal=risk, latency_ms=(time.perf_counter() - t0) * 1000,
        )
