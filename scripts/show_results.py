"""Print the results of a run in plain text: how good the classifier is, and how the whole agent does.

    python scripts/show_results.py                 # runs/full (the real run)
    python scripts/show_results.py quick           # runs/quick
    python scripts/show_results.py runs/archive/20261021-142700

Reads what the pipeline already wrote (reports/eval_results.json, models/*/meta.json,
thresholds.json, train_log.json, reports/multiturn_results.md); it does not run anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import _common  # noqa: F401
from _common import ROOT, RUNS

LEVELS = ["safe", "tense", "overwhelmed", "crisis"]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def pct(x) -> str:
    return "  -  " if x is None else f"{x:5.2f}"


def show_eval(run: Path, ev: dict) -> None:
    meta, res = ev["meta"], ev["results"]
    print(f"Test set: {meta['test_file']} ({meta['n_test']} messages)   classifier: {meta['classifier']}   LLM: {meta['llm']}")

    clf = res.get("classifier")
    if clf:
        print("\n1. The classifier on its own (mode 'classifier')")
        print(f"   accuracy {clf['exact_level_accuracy']:.2f}   macro-F1 {clf['macro_f1']:.2f}   "
              f"crisis recall {clf['crisis_recall']:.2f} (95% CI {clf['crisis_recall_ci95'][0]:.2f}-{clf['crisis_recall_ci95'][1]:.2f})")
        print("   level         precision  recall  F1     n")
        for name in LEVELS:
            c = clf["per_class"].get(name, {})
            print(f"   {name:<13} {c.get('precision', 0):9.2f} {c.get('recall', 0):7.2f} {c.get('f1-score', 0):5.2f} {int(c.get('support', 0)):5d}")
        print("   confusion (rows = true level, columns = predicted):")
        print("                 " + " ".join(f"{n[:5]:>6}" for n in LEVELS))
        for name, row in zip(LEVELS, clf["confusion"]):
            print(f"   {name:<13} " + " ".join(f"{v:6d}" for v in row))
    else:
        print("\n(no 'classifier' row: no trained classifier was found when evaluate.py ran)")

    print("\n2. All modes (same test set)")
    print("   mode              crisis recall (95% CI)  precision  style->crisis  tense->crisis  macro-F1  LLM calls")
    for mode, m in res.items():
        lo, hi = m["crisis_recall_ci95"]
        print(f"   {mode:<17} {m['crisis_recall']:5.2f} ({lo:.2f}-{hi:.2f})      {m['crisis_precision']:5.2f}"
              f"      {pct(m['style_false_crisis_rate'])}          {pct(m['tense_safe_false_crisis_rate'])}"
              f"      {m['macro_f1']:5.2f}    {m['llm_call_rate']:5.2f}")
    missed = run / "reports/missed_crises.csv"
    if missed.exists():
        n = max(0, len(missed.read_text(encoding="utf-8-sig").strip().splitlines()) - 1)
        print(f"   missed crises (all modes together): {n} rows in {missed.relative_to(ROOT) if missed.is_relative_to(ROOT) else missed}")


def show_model(d: Path) -> None:
    meta, th, log = load(d / "meta.json"), load(d / "thresholds.json"), load(d / "train_log.json")
    if not meta:
        return
    print(f"\n   {d.name}: {meta.get('base_model', meta['model_type'])}, trained on {meta['n_train']}, validated on {meta['n_val']}"
          + (f", best epoch {meta['best_epoch']}" if "best_epoch" in meta else ""))
    print(f"     validation: macro-F1 {meta['val_macro_f1']:.2f}, crisis recall {meta['val_crisis_recall']:.2f}")
    if log:
        print("     epochs: " + "  ".join(f"{r['epoch']}:{r['val_macro_f1']:.2f}" for r in log) + "   (epoch: val macro-F1)")
    if th:
        print(f"     calibration: temperature {th['temperature']:.2f}, ECE {th['ece_before']:.3f} -> {th['ece_after']:.3f}; "
              f"crisis >= {th['crisis_hi']:.2f} decides alone, {th['crisis_lo']:.2f}-{th['crisis_hi']:.2f} asks the LLM "
              f"(~{th['val_llm_call_rate']:.0%} of validation messages)")


def show_multiturn(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    total = re.search(r"(\d+)/(\d+) checks passed", text)
    failed = re.findall(r"^FAIL\s+(\S+):", text, re.M)
    if total:
        print(f"\n4. Multi-turn tests: {total.group(0)}" + (f"; failed: {', '.join(failed)}" if failed else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default="full", help="full | quick | a folder (default: full)")
    args = ap.parse_args()
    run = RUNS / args.run if args.run in ("full", "quick") else Path(args.run).resolve()
    print(f"=== {run}\n")

    ev = load(run / "reports/eval_results.json")
    if ev:
        show_eval(run, ev)
    else:
        print("No reports/eval_results.json yet: run the evaluate step first.")

    if any((run / "models" / m / "meta.json").exists() for m in ("transformer", "baseline")):
        print("\n3. Training-time numbers (validation set, not the test set)")
        for m in ("transformer", "baseline"):
            show_model(run / "models" / m)
    show_multiturn(run / "reports/multiturn_results.md")

    print("\nHow much to trust this:")
    if run.name == "quick":
        print("  - quick run: tiny training data and 32 test messages. It only shows that the steps work.")
    if ev:
        n3 = sum(ev["results"][next(iter(ev["results"]))]["confusion"][3])
        print(f"  - {n3} crisis messages in the test set; the 95% CI shows how far crisis recall could be off.")
        if ev["meta"]["test_file"] in ("test_set.jsonl", "test_set_draft.jsonl") and \
                sha(ROOT / "data/test_set.jsonl") == sha(ROOT / "data/test_set_draft.jsonl"):
            print("  - data/test_set.jsonl is still the draft. Final numbers need the team's rewritten test set.")
        if ev["meta"]["llm"].startswith("mock"):
            print("  - mock LLM: the rules+llm and full rows are not real results.")
        if ev["meta"].get("llm_errors"):
            print(f"  - LLM CALLS FAILED {ev['meta']['llm_errors']}: LM Studio was not reachable during evaluate, so the "
                  "rules+llm and full rows are NOT real results. Start LM Studio, then: python scripts/train.py --start-at evaluate")


if __name__ == "__main__":
    main()
