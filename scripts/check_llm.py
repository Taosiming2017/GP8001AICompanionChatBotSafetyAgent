"""Check the LLM path end to end: LM Studio reachable -> model loaded -> safety check -> companion reply.

    python scripts/check_llm.py                 # uses config.yaml
    python scripts/check_llm.py --provider mock # should always pass (offline stand-in)

Run this when the demo answers every message with "Sorry, I made a mistake..." or the patient page
says "Companion is resting". Each step prints PASS or FAIL with what to do.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback

import _common  # noqa: F401
import yaml
from _common import ROOT

from safety.llm_check import LLMCheckLayer
from safety.llm_providers import get_provider


def step(name: str):
    print(f"\n[{name}]")


def ok(msg: str):
    print(f"  PASS  {msg}")


def fail(msg: str, fix: str) -> None:
    print(f"  FAIL  {msg}\n        -> {fix}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default=None, help="override llm.provider in config.yaml")
    ap.add_argument("--model", default=None, help="override llm.model")
    ap.add_argument("--message", default="nobody would notice if i was gone", help="message for the safety check")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
    llm = cfg.get("llm") or {}
    comp = cfg.get("companion") or {}
    provider_name = args.provider or llm.get("provider") or "mock"
    model = args.model or (llm.get("model") if provider_name == llm.get("provider") else None)
    print(f"config.yaml: provider={provider_name} model={model or '(default)'} base_url={llm.get('base_url') or '(default)'}")

    step("1. connect")
    try:
        prov = get_provider(provider_name, model, timeout_s=comp.get("timeout_s", 120), base_url=llm.get("base_url"),
                            disable_thinking=comp.get("disable_thinking", True))
    except Exception as e:
        fail(f"cannot build the provider: {type(e).__name__}: {e}",
             "LM Studio: open it, load the model, Developer tab -> Start Server (port 1234). "
             "Cloud: set OPENAI_API_KEY / GEMINI_API_KEY and pip install openai / google-genai.")
        return 1
    ok(f"provider {prov.name}, model {prov.model}")

    if prov.name == "lmstudio":
        step("2. LM Studio model list")
        try:
            ids = prov.list_models()
        except Exception as e:
            fail(f"{prov.base_url}/models not reachable: {type(e).__name__}: {e}",
                 "In LM Studio go to the Developer tab and click Start Server. Check the port is 1234 "
                 "(or set llm.base_url in config.yaml). If 'Serve on local network' is on, localhost still works.")
            return 1
        if not ids:
            fail("server answers but no model is loaded", "Load the model in LM Studio (left panel -> Load).")
            return 1
        if prov.model not in ids:
            fail(f"config.yaml says '{prov.model}' but LM Studio has: {', '.join(ids)}",
                 f"Load that model, or set llm.model: {ids[0]} in config.yaml.")
            return 1
        ok(f"loaded: {', '.join(ids)}")

    step("3. safety check (JSON) on one message")
    t0 = time.perf_counter()
    res = LLMCheckLayer(prov).assess(args.message, None, "free_chat")
    dt = time.perf_counter() - t0
    if res.error:
        fail(f"{res.error}  ({dt:.1f} s)",
             "If it is a timeout: the model is too slow (GPU memory shared with the classifier? set classifier.device: cpu), "
             "or thinking mode is on in LM Studio (turn it off). If it is an HTTP 4xx: the server rejected the request; "
             "copy the message above into the issue.")
        return 1
    ok(f"level {int(res.level)} ({res.category or 'no category'}) in {dt:.1f} s: {res.reasons[0][:90]}")

    step("4. companion reply (plain chat)")
    msgs = [{"role": "system", "content": "You are Companion. Reply in one short, plain sentence.\nScenario id: free_chat"},
            {"role": "user", "content": "Hello. How are you today?"}]
    t0 = time.perf_counter()
    try:
        text = prov.chat(msgs, temperature=comp.get("temperature", 0.7), max_tokens=comp.get("max_tokens", 160))
    except Exception:
        fail(traceback.format_exc().strip().splitlines()[-1],
             "Same server as step 3, so this is usually a timeout: raise companion.timeout_s in config.yaml or use a smaller model.")
        return 1
    dt = time.perf_counter() - t0
    if not text.strip():
        fail(f"empty reply in {dt:.1f} s", "The model returned nothing (thinking block only?). Turn off thinking in LM Studio.")
        return 1
    ok(f"{dt:.1f} s: {text[:120]}")
    print("\nAll good. Start the demo with start_demo.bat.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
