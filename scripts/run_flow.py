"""The data-generation pipeline, end to end: make training data with an LLM, check it, then train.

    python scripts/run_flow.py                     # asks which mode
    python scripts/run_flow.py --mode quick        # tiny data, minutes; writes only to runs/quick/
    python scripts/run_flow.py --mode full         # the real run; writes to runs/full/
    python scripts/run_flow.py --mode full --start-at clean   # resume a full run from a step

You do NOT need this if the training data already exists (data/synthetic/): run scripts/train.py.
This script is for making or re-checking data with a local/cloud LLM:

    generate -> clean -> relabel -> review -> split      (data steps, this file)
    baseline -> transformer -> evaluate -> multiturn    (training steps, delegated to scripts/train.py)

quick  5 messages per label + the 126 seeds, relabel check on 20 items, 32 test messages.
       Writes only to runs/quick/, which is wiped at every start. Its numbers only show that the flow works.
full   every step at full size, written to runs/full/. Generation resumes where it stopped.
       It pauses when you need to act (switch models in LM Studio, review runs/full/data/review/to_review.csv,
       free GPU memory before training). Before it overwrites trained models, reports or the
       review sheet, it copies the old ones to runs/archive/<date-time>/.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime

import _common  # noqa: F401
import yaml
from _common import FULL, ROOT, RUNS, read_jsonl, write_jsonl

PY = sys.executable
QUICK = RUNS / "quick"     # same layout as runs/full/, tiny data
ARCHIVE = RUNS / "archive"
DATA_STEPS = ["generate", "clean", "relabel", "review", "split"]
TRAIN_STEPS = ["baseline", "transformer", "evaluate", "multiturn"]
FULL_STEPS = DATA_STEPS + TRAIN_STEPS


class Flow:
    def __init__(self, pauses: bool):
        self.log = []
        self.pauses = pauses

    def run(self, title, script, args):
        n = len(self.log) + 1
        shown = " ".join(str(a) for a in args)
        print(f"\n=== Step {n}: {title}\n    python scripts/{script} {shown}", flush=True)
        t0 = time.time()
        code = subprocess.run([PY, str(ROOT / "scripts" / script), *[str(a) for a in args]], cwd=ROOT).returncode
        dt = time.time() - t0
        self.log.append((n, title, "ok" if code == 0 else f"FAILED (exit {code})", dt))
        if code != 0:
            print(f"\nStep {n} failed. Read the error above, fix it, then run again"
                  " (in full mode, add --start-at <step> to skip the steps that already worked).")
            self.summary()
            sys.exit(code)
        print(f"    done in {dt:.0f}s", flush=True)

    def pause(self, msg):
        print(f"\n>>> {msg}", flush=True)
        if self.pauses:
            input(">>> Press Enter when ready (Ctrl+C to stop) ")

    def summary(self):
        print("\n" + "=" * 64 + "\nSummary")
        for n, title, status, dt in self.log:
            print(f"  {n:>2}. {title:<42} {status:<18} {dt:6.0f}s")
        print(f"  total {sum(r[3] for r in self.log) / 60:.1f} min")


# ---------------------------------------------------------------------------- LM Studio helpers
def lmstudio_ids():
    from safety.llm_providers import LMStudioProvider

    return LMStudioProvider(model="probe", timeout_s=10).list_models()


def ensure_lmstudio(flow, model, role):
    """Check the server answers; return the model id to use."""
    for attempt in range(3):
        try:
            ids = lmstudio_ids()
        except Exception as e:
            print(f"Cannot reach the LM Studio server ({type(e).__name__}).")
            if attempt == 2 or not flow.pauses:
                sys.exit("Open LM Studio, load a model, then Developer tab -> Start server, and run again.")
            flow.pause("Open LM Studio, load the model, and turn on Developer -> Start server.")
            continue
        if not ids:
            if attempt == 2 or not flow.pauses:
                sys.exit("The LM Studio server is running but no model is loaded.")
            flow.pause("The LM Studio server is running but no model is loaded. Load one.")
            continue
        chosen = model or ids[0]
        if model and model not in ids:
            print(f"  note: '{model}' is not in the server's model list {ids}. Check the id.")
        print(f"LM Studio OK. {role}: {chosen}")
        return chosen
    sys.exit("LM Studio not ready.")


def train_args(args, provider, model, extra):
    """Options passed through to scripts/train.py (it does its own LM Studio pauses)."""
    out = ["--llm-provider", provider, "--no-archive"] + (["--llm-model", model] if model else [])
    if args.no_pause:
        out.append("--no-pause")
    return out + extra


# ---------------------------------------------------------------------------- quick mode
def sample_test(n_per_level, seed=0):
    src = ROOT / "data/test_set.jsonl"
    if not src.exists():
        src = ROOT / "data/test_set_draft.jsonl"
    by = defaultdict(list)
    for r in read_jsonl(src):
        by[r["level"]].append(r)
    rng = random.Random(seed)
    out = []
    for lvl in sorted(by):
        out += rng.sample(by[lvl], min(n_per_level, len(by[lvl])))
    return src.name, out


def run_quick(args, cfg):
    flow = Flow(pauses=not args.no_pause)
    lm = args.provider == "lmstudio"
    model = args.model or cfg.get("llm", {}).get("model")
    if lm:
        model = ensure_lmstudio(flow, model, "model")
    elif args.provider == "mock":
        model = None
    margs = ["--model", model] if model else []

    if QUICK.exists():
        shutil.rmtree(QUICK)
    (QUICK / "reports").mkdir(parents=True)
    qd = QUICK / "data"
    test_src, test = sample_test(args.test_per_level)
    write_jsonl(qd / "test_quick.jsonl", test)
    print(f"Quick run. Outputs go to {QUICK}. Test set: {len(test)} messages sampled from data/{test_src}")

    raw = qd / "raw/synthetic_raw.jsonl"
    if args.no_generate or args.provider == "mock":
        print("\n(generation skipped: training on the 126 seed examples only)")
    else:
        flow.run("generate 1 small batch per label", "generate_synthetic.py",
                 ["--provider", args.provider, *margs, "--out", raw, "--calls-per-subtype", 1,
                  "--items-per-call", args.items_per_call])
        if not raw.exists():
            print("    note: nothing was generated (refusals or timeouts); continuing with the seeds only")

    inputs = ["--inputs", raw, "--include-seeds"] if raw.exists() else ["--inputs", ROOT / "data/seeds.jsonl"]
    flow.run("clean and de-duplicate", "clean_dedup.py",
             [*inputs, "--test", qd / "test_quick.jsonl", "--out", qd / "interim/clean.jsonl", "--reports", QUICK / "reports"])
    flow.run(f"relabel check ({args.relabel_sample} items)", "relabel_check.py",
             ["--provider", args.provider, *margs, "--inp", qd / "interim/clean.jsonl", "--out", qd / "interim/relabeled_sample.jsonl",
              "--sample", args.relabel_sample, "--workers", 1, "--review-dir", qd / "review", "--reports", QUICK / "reports"])
    flow.run("split train / validation", "split.py", ["--inp", qd / "interim/clean.jsonl", "--outdir", qd / "processed"])

    extra = ["--train", qd / "processed/train.jsonl", "--val", qd / "processed/val.jsonl", "--test", qd / "test_quick.jsonl",
             "--out", QUICK, "--epochs", args.epochs, "--lr", 5e-5] + (["--skip-transformer"] if args.skip_transformer else [])
    flow.run("train, calibrate, evaluate, multi-turn (scripts/train.py)", "train.py", train_args(args, args.provider, model, extra))

    flow.summary()
    print_results(QUICK / "reports")
    print("Quick-mode numbers only show that the flow works; they are not results.")


# ---------------------------------------------------------------------------- full mode
def archive_outputs():
    dest = ARCHIVE / datetime.now().strftime("%Y%m%d-%H%M%S")
    copied = []
    for rel in ("models", "reports", "data/review"):
        src = FULL / rel
        if src.exists() and any(p.name != ".gitkeep" for p in src.iterdir()):
            shutil.copytree(src, dest / rel, ignore=shutil.ignore_patterns(".gitkeep"))
            copied.append(rel)
    if copied:
        print(f"Previous outputs ({', '.join(copied)}) copied to {dest}")


def run_full(args, cfg):
    flow = Flow(pauses=not args.no_pause)
    lm = args.provider == "lmstudio"
    steps = FULL_STEPS[FULL_STEPS.index(args.start_at): FULL_STEPS.index(args.stop_after) + 1]
    model = args.model or cfg.get("llm", {}).get("model")
    if args.provider == "mock":
        sys.exit("Full mode needs a real model (lmstudio, openai or gemini). Use --mode quick --provider mock to test plumbing.")

    print("Full run. Steps: " + " -> ".join(steps))
    print(f"Outputs go to {FULL} (data/, models/, reports/: the folder the demo and the slides use).")
    if not (ROOT / "data/test_set.jsonl").exists():
        print("note: data/test_set.jsonl not found, so the draft test set will be used. "
              "Final numbers must come from the team's rewritten test set.")
    relabel_model = args.relabel_model
    if "relabel" in steps and lm and not relabel_model and flow.pauses:
        relabel_model = input("\nModel B id for the relabel check (another model family; Enter = same as model A): ").strip() or None
    if "relabel" in steps and (relabel_model is None or relabel_model == model):
        print("note: the relabel check will use the same model that generated the data; agreement will look too good.")
    flow.pause("This can take hours. Ready to start?")
    if not args.no_archive and set(steps) & {"relabel", "review", *TRAIN_STEPS}:
        archive_outputs()

    P = args.provider
    margs = ["--model", model] if model else []
    if "generate" in steps:
        if lm:
            flow.pause(f"LM Studio: load model A ({model or 'the generator'}) and turn on the server.")
            model = ensure_lmstudio(flow, model, "model A")
            margs = ["--model", model]
        gen = ["--provider", P, *margs]
        if args.items_per_call_full:
            gen += ["--items-per-call", args.items_per_call_full]
        flow.run("generate training data (resumable)", "generate_synthetic.py", gen)
    if "clean" in steps:
        flow.run("clean and de-duplicate", "clean_dedup.py", ["--include-seeds"])
    if "relabel" in steps:
        if lm:
            flow.pause(f"LM Studio: eject model A and load model B ({relabel_model or 'same as A'}).")
            relabel_model = ensure_lmstudio(flow, relabel_model or model, "model B")
        flow.run("relabel check with model B", "relabel_check.py",
                 ["--provider", P, *(["--model", relabel_model] if relabel_model else []), "--workers", 2])
    if "review" in steps:
        sheet = FULL / "data/review/to_review.csv"
        if sheet.exists():
            flow.pause(f"Open {sheet} in Excel. Fill the final_level column (0-3, or 'drop'), "
                       "save as 'CSV UTF-8', and close Excel. Rows left blank keep their label.")
            flow.run("apply the team review", "relabel_check.py", ["--apply-review", sheet])
        else:
            print("\n(no review sheet found; run the relabel step first)")
    if "split" in steps:
        flow.run("split train / validation", "split.py", [])

    train_steps = [s for s in steps if s in TRAIN_STEPS]
    if train_steps:
        extra = ["--train", FULL / "data/processed/train.jsonl", "--val", FULL / "data/processed/val.jsonl", "--out", FULL,
                 "--start-at", train_steps[0], "--stop-after", train_steps[-1]]
        flow.run("train, calibrate, evaluate, multi-turn (scripts/train.py)", "train.py", train_args(args, P, model, extra))

    flow.summary()
    if "evaluate" in steps:
        print_results(FULL / "reports")


def print_results(reports):
    path = reports / "eval_results.json"
    if not path.exists():
        return
    res = json.loads(path.read_text(encoding="utf-8"))["results"]
    if "full" in res:
        m = res["full"]
        print(f"\nfull mode on {m['n']} test messages: crisis recall {m['crisis_recall']:.2f}, "
              f"macro-F1 {m['macro_f1']:.2f}, LLM call rate {m['llm_call_rate']:.2f}")
    print(f"Details: {reports / 'eval_report.md'} and {reports / 'multiturn_results.md'}; or: python scripts/show_results.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["quick", "full"], help="if omitted, you are asked")
    ap.add_argument("--provider", default="lmstudio", help="lmstudio | openai | gemini | mock (quick only)")
    ap.add_argument("--model", default=None, help="model A id (default: llm.model in config.yaml)")
    ap.add_argument("--no-pause", action="store_true", help="never wait for Enter (only if LM Studio needs no changes)")
    g = ap.add_argument_group("quick mode")
    g.add_argument("--no-generate", action="store_true", help="train on the 126 seed examples only")
    g.add_argument("--items-per-call", type=int, default=5, help="messages per label (default 5)")
    g.add_argument("--test-per-level", type=int, default=8, help="test messages per level (default 8)")
    g.add_argument("--relabel-sample", type=int, default=20)
    g.add_argument("--skip-transformer", action="store_true", help="train only the TF-IDF baseline")
    g.add_argument("--epochs", type=int, default=8)
    f = ap.add_argument_group("full mode")
    f.add_argument("--relabel-model", default=None, help="model B id for the relabel check (another family)")
    f.add_argument("--start-at", choices=FULL_STEPS, default=FULL_STEPS[0])
    f.add_argument("--stop-after", choices=FULL_STEPS, default=FULL_STEPS[-1])
    f.add_argument("--items-per-call-full", type=int, default=None,
                   help="messages per generation call (default 20); keep it the same when resuming")
    f.add_argument("--no-archive", action="store_true", help="do not copy old models/reports/review to runs/archive")
    args = ap.parse_args()

    if not args.mode:
        print("Which run?\n"
              "  1  quick : tiny data, checks every step in minutes; writes only to runs/quick\n"
              "  2  full  : the real run, can take hours; writes to runs/full")
        args.mode = {"1": "quick", "2": "full", "quick": "quick", "full": "full"}.get(input("Type 1 or 2: ").strip().lower())
        if not args.mode:
            sys.exit("No mode chosen.")

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    (run_quick if args.mode == "quick" else run_full)(args, cfg)


if __name__ == "__main__":
    main()
