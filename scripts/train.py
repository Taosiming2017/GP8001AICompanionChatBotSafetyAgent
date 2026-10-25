"""Train, calibrate and evaluate the classifier from a ready-made train/val split, in one command.

    python scripts/train.py                                   # data/synthetic -> runs/full, LLM from config.yaml
    python scripts/train.py --llm-provider mock               # offline: skip LM Studio (full-mode numbers not real)
    python scripts/train.py --skip-transformer                # TF-IDF baseline only (CPU, a minute)
    python scripts/train.py --start-at evaluate               # models exist already; only evaluate + multi-turn
    python scripts/train.py --epochs 6 --lr 3e-5 --crisis-weight 2.5

Steps (names for --start-at / --stop-after):
    baseline     TF-IDF + logistic regression, then calibrate       -> <out>/models/baseline
    transformer  fine-tune DistilRoBERTa, then calibrate            -> <out>/models/transformer
    evaluate     all 5 modes on data/test_set.jsonl                 -> <out>/reports/eval_report.md
    multiturn    scripted conversations through the whole agent     -> <out>/reports/multiturn_results.md
    (then scripts/plot_results.py draws the slide figures into <out>/reports/figures/)

The GPU is shared with LM Studio: before `transformer` the script checks free GPU memory and asks
you to eject the model in LM Studio if there is not enough; before `evaluate` with LM Studio it
asks you to load it again. Old models and reports in <out> are copied to runs/archive/<time>/ first.
scripts/run_flow.py (the data-generation pipeline) calls this script for its training steps.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import _common  # noqa: F401
import yaml
from _common import FULL, ROOT, RUNS

PY = sys.executable
STEPS = ["baseline", "transformer", "evaluate", "multiturn"]
DATA = ROOT / "data" / "synthetic"


def run(title: str, script: str, args: list, log: list) -> None:
    shown = " ".join(str(a) for a in args)
    print(f"\n=== {title}\n    python scripts/{script} {shown}", flush=True)
    t0 = time.time()
    code = subprocess.run([PY, str(ROOT / "scripts" / script), *[str(a) for a in args]], cwd=ROOT).returncode
    log.append((title, "ok" if code == 0 else f"FAILED (exit {code})", time.time() - t0))
    if code != 0:
        summary(log)
        sys.exit(f"\n'{title}' failed. Read the error above, fix it, then run again with --start-at <step>.")
    print(f"    done in {time.time() - t0:.0f}s", flush=True)


def summary(log: list) -> None:
    print("\n" + "=" * 64 + "\nSummary")
    for title, status, dt in log:
        print(f"  {title:<40} {status:<18} {dt:6.0f}s")
    print(f"  total {sum(r[2] for r in log) / 60:.1f} min")


def pause(msg: str, enabled: bool) -> None:
    print(f"\n>>> {msg}", flush=True)
    if enabled:
        input(">>> Press Enter when ready (Ctrl+C to stop) ")


def gpu_free_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _ = torch.cuda.mem_get_info()
        return free / 1e9
    except Exception:
        return None


def ensure_lmstudio(model: str | None, cfg: dict, pauses: bool) -> None:
    """Do not start evaluating until LM Studio answers and lists the model (otherwise every LLM call fails
    silently, the fail-safe decides, and the full / rules+llm rows look real but are not)."""
    from safety.llm_providers import LMStudioProvider

    base_url = (cfg.get("llm") or {}).get("base_url")
    for attempt in range(20):
        try:
            ids = LMStudioProvider(model="probe", timeout_s=5, base_url=base_url).list_models()
        except Exception as e:
            problem = f"LM Studio server not reachable ({type(e).__name__})"
        else:
            if not ids:
                problem = "LM Studio server is running but no model is loaded"
            elif model and model not in ids:
                problem = f"LM Studio has {ids}, not '{model}' (config.yaml llm.model)"
            else:
                print(f"LM Studio OK: {model or ids[0]}")
                return
        if not pauses:
            sys.exit(f"{problem}. Start LM Studio (load the model, Developer -> Start Server) or use --llm-provider mock.")
        pause(f"{problem}. Load the model ({model or 'any'}) in LM Studio and click Developer -> Start Server. "
              "The classifier will run on the CPU if the GPU is full; that is fine.", True)
    sys.exit("LM Studio still not ready.")


def archive(out: Path) -> None:
    dest = RUNS / "archive" / datetime.now().strftime("%Y%m%d-%H%M%S")
    copied = []
    for rel in ("models", "reports"):
        src = out / rel
        if src.exists() and any(src.iterdir()):
            shutil.copytree(src, dest / rel)
            copied.append(rel)
    if copied:
        print(f"Previous {', '.join(copied)} copied to {dest}")


def write_run_config(out: Path, provider: str, model: str | None, cfg: dict) -> Path:
    """A config.yaml inside the run folder that points at this run's models (for the multi-turn tests)."""
    run_cfg = dict(cfg)
    run_cfg["classifier"] = {**(cfg.get("classifier") or {}), "model_dir": "models/transformer", "fallback_dir": "models/baseline"}
    run_cfg["llm"] = {**(cfg.get("llm") or {}), "provider": provider, "model": model,
                      "timeout_s": 300 if provider == "lmstudio" else (cfg.get("llm") or {}).get("timeout_s", 20)}
    path = out / "config.yaml"
    path.write_text(yaml.safe_dump(run_cfg, sort_keys=False), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=str(DATA / "train.jsonl"))
    ap.add_argument("--val", default=str(DATA / "val.jsonl"))
    ap.add_argument("--test", default=str(ROOT / "data/test_set.jsonl"))
    ap.add_argument("--out", default=str(FULL), help="run folder (default runs/full, the one config.yaml loads)")
    ap.add_argument("--start-at", choices=STEPS, default=STEPS[0])
    ap.add_argument("--stop-after", choices=STEPS, default=STEPS[-1])
    ap.add_argument("--skip-transformer", action="store_true", help="baseline only")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--crisis-weight", type=float, default=2.0)
    ap.add_argument("--model", default="distilroberta-base", help="base model for the transformer")
    ap.add_argument("--llm-provider", default=None, help="for evaluate/multiturn: lmstudio | openai | gemini | mock (default: config.yaml)")
    ap.add_argument("--llm-model", default=None, help="model id (default: llm.model in config.yaml for that provider)")
    ap.add_argument("--no-pause", action="store_true", help="never wait for Enter (GPU already free / model already loaded)")
    ap.add_argument("--no-archive", action="store_true", help="do not copy old models/reports to runs/archive")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
    llm_cfg = cfg.get("llm") or {}
    provider = args.llm_provider or llm_cfg.get("provider") or "mock"
    model = args.llm_model or (llm_cfg.get("model") if provider == llm_cfg.get("provider") else None)
    out = Path(args.out).resolve()
    steps = STEPS[STEPS.index(args.start_at): STEPS.index(args.stop_after) + 1]
    if args.skip_transformer:
        steps = [s for s in steps if s != "transformer"]
    for p in (args.train, args.val):
        if not Path(p).exists():
            sys.exit(f"missing {p}. Put train.jsonl / val.jsonl in data/synthetic/ or pass --train/--val.")
    if not Path(args.test).exists():
        sys.exit(f"missing test set {args.test}")

    pauses = not args.no_pause
    print(f"Steps: {' -> '.join(steps)}\nTrain: {args.train}\nVal:   {args.val}\nTest:  {args.test}\nOut:   {out}"
          f"\nLLM for evaluate/multiturn: {provider}/{model or 'default'}")
    if provider.startswith("mock"):
        print("note: mock LLM -> the rules+llm and full rows of the evaluation are NOT real results")
    if not args.no_archive and set(steps) & {"baseline", "transformer", "evaluate", "multiturn"}:
        archive(out)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    log: list = []
    tv = ["--train", args.train, "--val", args.val]

    if "baseline" in steps:
        run("train TF-IDF baseline", "train_baseline.py", [*tv, "--out", out / "models/baseline"], log)
        run("calibrate baseline", "calibrate.py", ["--model-dir", out / "models/baseline"], log)

    if "transformer" in steps:
        free = gpu_free_gb()
        if free is not None and free < 3.0:
            pause(f"Only {free:.1f} GB of GPU memory is free (LM Studio is probably holding the model). "
                  "Eject the model in LM Studio (or training runs much slower / fails), then continue.", pauses)
        elif free is not None:
            print(f"GPU: {free:.1f} GB free")
        run("fine-tune DistilRoBERTa", "train_transformer.py",
            [*tv, "--out", out / "models/transformer", "--model", args.model, "--epochs", args.epochs,
             "--lr", args.lr, "--crisis-weight", args.crisis_weight], log)
        run("calibrate DistilRoBERTa", "calibrate.py", ["--model-dir", out / "models/transformer"], log)

    model_dir = out / ("models/transformer" if (out / "models/transformer/meta.json").exists() else "models/baseline")
    run_cfg = write_run_config(out, provider, model, cfg)
    if provider == "lmstudio" and set(steps) & {"evaluate", "multiturn"}:
        ensure_lmstudio(model, cfg, pauses)
        run("evaluate on the test set (5 modes)", "evaluate.py",
            ["--test", args.test, "--model-dir", model_dir, "--llm-provider", provider,
             *(["--llm-model", model] if model else []), "--out", out / "reports"], log)

    if "multiturn" in steps:
        run("multi-turn tests", "run_multiturn_tests.py",
            ["--config", run_cfg, "--out", out / "reports/multiturn_results.md"], log)

    if "evaluate" in steps:   # slide figures; a failure here must not fail the run
        code = subprocess.run([PY, str(ROOT / "scripts/plot_results.py"), str(out)], cwd=ROOT).returncode
        log.append(("figures (scripts/plot_results.py)", "ok" if code == 0 else f"FAILED (exit {code})", 0.0))

    summary(log)
    print(f"\nRead the results:  python scripts/show_results.py" + ("" if out == FULL.resolve() else f" {out}"))
    print(f"Slide figures:     {out / 'reports' / 'figures'}  (captions in figures.md)")
    print(f"Try the model:     python scripts/predict.py --model-dir {model_dir} \"too loud, stop\" \"i want to die\"")
    if out == FULL.resolve():
        print("The demo (start_demo.bat) now loads this classifier automatically.")


if __name__ == "__main__":
    main()
