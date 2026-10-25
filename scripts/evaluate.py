"""Step 7 - evaluate the safety agent on the team-written test set, with ablations.

    python scripts/evaluate.py --model-dir runs/full/models/transformer --llm-provider openai
    python scripts/evaluate.py --model-dir runs/full/models/baseline --llm-provider mock   # offline (full-mode numbers not meaningful)

Modes compared (same test set):
    rules              keyword rules only (soft cues fail closed -> crisis)
    classifier         fine-tuned classifier only
    rules+classifier   no LLM
    rules+llm          the proposal's original demo design (no classifier)
    full               rules -> classifier -> LLM (the design we ship)

Key numbers for the slides
    crisis recall (+95% Wilson CI)      - the most important number
    false-crisis rate on autistic-style messages (blunt / idiom / special interest)
    false-crisis rate on tense-but-safe messages (gold level 1-2)
    macro-F1 over the 4 levels, LLM call rate, latency
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import _common  # noqa: F401
import numpy as np
from _common import FULL, ROOT, read_jsonl, wilson_ci
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from safety.agent import SafetyAgent
from safety.classifier import load_classifier
from safety.llm_providers import get_provider

STYLE_SUBTYPES = {"blunt_literal", "idiom_negative", "special_interest", "meta_feedback"}
COST = np.array([[0, 1, 2, 3], [3, 0, 1, 2], [6, 3, 0, 1], [9, 6, 3, 0]])  # rows gold, cols pred; misses cost 3x


class CachingProvider:
    """Wraps a provider so rules+llm and full modes do not pay twice for the same message."""

    def __init__(self, inner):
        self.inner, self.cache, self.calls = inner, {}, 0
        self.name, self.model = inner.name, inner.model

    def complete_json(self, system, user, temperature=0.0, max_tokens=400, schema=None):
        key = (system, user)
        if key not in self.cache:
            self.calls += 1
            self.cache[key] = self.inner.complete_json(system, user, temperature, max_tokens, schema=schema)
        return self.cache[key]


def run_mode(agent, test):
    preds, cats, llm_flags, lat, layers_log, llm_errors = [], [], [], [], [], 0
    for r in test:
        t0 = time.perf_counter()
        a = agent.assess_message(r["text"], r.get("context"), r.get("scenario"))
        lat.append((time.perf_counter() - t0) * 1000)
        preds.append(int(a.level))
        cats.append(a.category)
        llm_flags.append(a.llm_called)
        llm_errors += any(l.layer == "llm" and l.error for l in a.layers)
        layers_log.append({"layers": [(l.layer, None if l.level is None else int(l.level)) for l in a.layers], "notes": a.notes})
    return np.array(preds), cats, np.array(llm_flags), np.array(lat), layers_log, llm_errors


def metrics(test, y, p, llm_flags, lat):
    sub = np.array([r["subtype"] for r in test])
    is3, n3 = y == 3, int((y == 3).sum())
    tp3 = int(((p == 3) & is3).sum())
    style = np.isin(sub, list(STYLE_SUBTYPES))
    tense_safe = np.isin(y, [1, 2])
    rec_ci = wilson_ci(tp3, n3)
    return {
        "n": int(len(y)),
        "crisis_recall": tp3 / max(1, n3), "crisis_recall_ci95": rec_ci, "crisis_missed": n3 - tp3,
        "crisis_precision": tp3 / max(1, int((p == 3).sum())),
        "false_crisis_rate_noncrisis": float(((p == 3) & ~is3).sum() / max(1, (~is3).sum())),
        "style_any_flag_rate": float((p[style] >= 1).mean()) if style.any() else None,
        "style_false_crisis_rate": float((p[style] == 3).mean()) if style.any() else None,
        "tense_safe_false_crisis_rate": float((p[tense_safe] == 3).mean()) if tense_safe.any() else None,
        "macro_f1": float(f1_score(y, p, average="macro", labels=[0, 1, 2, 3])),
        "exact_level_accuracy": float((y == p).mean()),
        "under_escalation_rate": float((p < y).mean()),
        "severity_weighted_cost": float(COST[y, p].mean()),
        "llm_call_rate": float(llm_flags.mean()),
        "latency_ms_mean": float(lat.mean()), "latency_ms_p95": float(np.percentile(lat, 95)),
        "confusion": confusion_matrix(y, p, labels=[0, 1, 2, 3]).tolist(),
        "per_class": classification_report(y, p, labels=[0, 1, 2, 3], target_names=["safe", "tense", "overwhelmed", "crisis"],
                                           output_dict=True, zero_division=0),
    }


def plot_confusion(cm, title, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    names = ["safe", "tense", "overwh.", "crisis"]
    ax.set_xticks(range(4), names)
    ax.set_yticks(range(4), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=str(ROOT / "data/test_set.jsonl"))
    ap.add_argument("--model-dir", default=str(FULL / "models/transformer"))
    ap.add_argument("--llm-provider", default="mock")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--modes", default="rules,classifier,rules+classifier,rules+llm,full")
    ap.add_argument("--out", default=str(FULL / "reports"))
    args = ap.parse_args()

    test_path = Path(args.test)
    if not test_path.exists():
        test_path = ROOT / "data/test_set_draft.jsonl"
        print(f"note: using {test_path.name} (team-reviewed data/test_set.jsonl not found)")
    test = read_jsonl(test_path)
    y = np.array([r["level"] for r in test])

    clf = load_classifier(args.model_dir)
    provider = CachingProvider(get_provider(args.llm_provider, args.llm_model,
                                            timeout_s=300 if args.llm_provider == "lmstudio" else 20))
    if args.llm_provider.startswith("mock"):
        print("WARNING: mock LLM - numbers for rules+llm and full are NOT real results")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results, missed_rows, preds = {}, [], {}
    for mode in args.modes.split(","):
        needs_clf = mode in ("classifier", "rules+classifier", "full")
        if needs_clf and clf is None:
            print(f"skip {mode}: no trained classifier in {args.model_dir}")
            continue
        agent = SafetyAgent(classifier=clf if needs_clf else None,
                            llm_provider=provider if mode in ("rules+llm", "full") else None, mode=mode)
        p, cats, llm_flags, lat, log, llm_errors = run_mode(agent, test)
        m = metrics(test, y, p, llm_flags, lat)
        m["llm_errors"] = int(llm_errors)
        if llm_errors:
            first = next(n for lg in log for n in lg["notes"] if n.startswith("LLM failed"))
            print(f"\nWARNING: {mode}: {llm_errors} of {int(llm_flags.sum())} LLM calls FAILED, so the fail-safe decided "
                  f"instead of the model and this row is NOT a real result.\n         {first[:160]}\n"
                  "         Is LM Studio running with the model loaded? Check with: python scripts/check_llm.py\n")
        m["crisis_recall_by_subtype"] = {   # subtype -> [caught, total], for the per-subtype figure
            st: [int(((p == 3) & (y == 3) & (sub == st)).sum()), int(((y == 3) & (sub == st)).sum())]
            for sub in [np.array([r["subtype"] for r in test])] for st in sorted(set(sub[y == 3]))}
        results[mode] = m
        preds[mode] = [int(v) for v in p]
        plot_confusion(m["confusion"], f"{mode}  (n={len(y)})", out / f"confusion_{mode.replace('+', '_')}.png")
        for r, pred, lg in zip(test, p, log):
            if r["level"] == 3 and pred < 3:
                missed_rows.append([mode, r["id"], r["subtype"], pred, r["text"], json.dumps(lg)])
        lo, hi = m["crisis_recall_ci95"]
        print(f"{mode:<17} crisis recall {m['crisis_recall']:.2f} [{lo:.2f}-{hi:.2f}]  "
              f"style false-crisis {m['style_false_crisis_rate']:.2f}  tense->crisis {m['tense_safe_false_crisis_rate']:.2f}  "
              f"macro-F1 {m['macro_f1']:.2f}  LLM calls {m['llm_call_rate']:.2f}")

    meta = {"test_file": test_path.name, "n_test": len(test), "classifier": args.model_dir if clf else None,
            "llm": f"{provider.name}/{provider.model}", "llm_api_calls": provider.calls,
            "llm_errors": {m: r["llm_errors"] for m, r in results.items() if r.get("llm_errors")}}
    (out / "eval_results.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8")
    with open(out / "missed_crises.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["mode", "id", "subtype", "predicted_level", "text", "layers"])
        w.writerows(missed_rows)
    with open(out / "predictions.csv", "w", newline="", encoding="utf-8-sig") as f:  # every item, every mode
        w = csv.writer(f)
        w.writerow(["id", "subtype", "scenario", "gold", *[f"pred_{m}" for m in preds], "text"])
        for i, r in enumerate(test):
            w.writerow([r["id"], r["subtype"], r.get("scenario", ""), r["level"], *[preds[m][i] for m in preds], r["text"]])

    lines = ["# Safety agent evaluation", "", f"Test file: `{meta['test_file']}` (n={len(test)}); classifier: `{meta['classifier']}`; LLM: `{meta['llm']}`", ""]
    if meta["llm_errors"]:
        lines += [f"> **WARNING: LLM calls failed** ({meta['llm_errors']}). The fail-safe decided those messages, so the "
                  "rules+llm and full rows are NOT real results. Start LM Studio and re-run the evaluate step.", ""]
    lines += [
             "| mode | crisis recall (95% CI) | crisis precision | style false-crisis | tense-safe false-crisis | macro-F1 | weighted cost | LLM call rate | mean latency (ms) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for mode, m in results.items():
        lo, hi = m["crisis_recall_ci95"]
        lines.append(f"| {mode} | {m['crisis_recall']:.2f} ({lo:.2f}-{hi:.2f}) | {m['crisis_precision']:.2f} | "
                     f"{m['style_false_crisis_rate']:.2f} | {m['tense_safe_false_crisis_rate']:.2f} | {m['macro_f1']:.2f} | "
                     f"{m['severity_weighted_cost']:.2f} | {m['llm_call_rate']:.2f} | {m['latency_ms_mean']:.0f} |")
    lines += ["", "Missed crises are listed in `missed_crises.csv`; review every one by hand. "
              "All predictions: `predictions.csv`. Slide figures: `python scripts/plot_results.py`."]
    (out / "eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out / 'eval_report.md'}")


if __name__ == "__main__":
    main()
