"""Ask the trained classifier (and, with --agent, the whole safety agent) about some messages.

    python scripts/predict.py "I'd kill for chicken rice" "i want to die"
    python scripts/predict.py --model-dir runs/quick/models/transformer "too loud, stop"
    python scripts/predict.py --file my_messages.txt                 # one message per line
    python scripts/predict.py --agent "nobody would notice if i was gone"   # rules + classifier + LLM, as in the app

Without --model-dir it uses the classifier that config.yaml points to (runs/full/models/...).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401
import yaml
from _common import ROOT

from safety.classifier import ClassifierLayer, load_classifier
from safety.schema import LABELS


def default_model_dir() -> Path | None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")).get("classifier", {})
    for key in ("model_dir", "fallback_dir"):
        d = cfg.get(key)
        if d and (ROOT / d / "meta.json").exists():
            return ROOT / d
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("messages", nargs="*")
    ap.add_argument("--file", help="text file, one message per line")
    ap.add_argument("--model-dir", default=None, help="a trained model folder (has meta.json)")
    ap.add_argument("--context", default=None, help="the companion's previous line")
    ap.add_argument("--scenario", default="free_chat", help="hawker_centre | new_colleague | job_interview | free_chat")
    ap.add_argument("--agent", action="store_true", help="also run the whole agent (uses the LLM in config.yaml)")
    ap.add_argument("--provider", default=None, help="with --agent: override llm.provider (e.g. mock)")
    args = ap.parse_args()

    msgs = list(args.messages)
    if args.file:
        msgs += [l.strip() for l in Path(args.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not msgs:
        ap.error("give at least one message (or --file)")

    model_dir = Path(args.model_dir).resolve() if args.model_dir else default_model_dir()
    clf = load_classifier(model_dir) if model_dir else None
    if clf is None:
        raise SystemExit("No trained classifier found. Train one (scripts/run_flow.py) or pass --model-dir.")
    layer = ClassifierLayer(clf)
    print(f"classifier: {model_dir}  (crisis >= {layer.th['crisis_hi']:.2f} decides alone, "
          f"{layer.th['crisis_lo']:.2f}-{layer.th['crisis_hi']:.2f} asks the LLM)\n")

    agent = None
    if args.agent:
        from safety.agent import SafetyAgent

        agent = SafetyAgent.from_config(ROOT / "config.yaml", provider=args.provider)

    print(f"{'message':<46} " + " ".join(f"{n[:5]:>6}" for n in LABELS) + "  classifier" + ("   agent" if agent else ""))
    for m in msgs:
        r = layer.assess(m, args.context, args.scenario)
        probs = " ".join(f"{r.probs[n]:6.2f}" for n in LABELS)
        tag = f"{int(r.level)} {LABELS[int(r.level)]}" + (" (asks LLM)" if r.wants_llm else "")
        line = f"{(m[:43] + '...') if len(m) > 46 else m:<46} {probs}  {tag}"
        if agent:
            a = agent.assess_message(m, args.context, args.scenario)
            line += f"   {int(a.level)} {LABELS[int(a.level)]}" + (f" [{a.notes[-1]}]" if a.notes else "")
        print(line)


if __name__ == "__main__":
    main()
