"""Local demo server: two pages, one small JSON API, nothing to install beyond the agent itself.

    python demo/server.py                    # LLM from config.yaml (LM Studio by default); opens both pages
    python demo/server.py --provider mock    # offline: canned companion + rough safety stand-in
    python demo/server.py --no-browser --port 8800

Pages (served from demo/web/):
    http://localhost:8765/patient   the user's chat. Only text and buttons; no levels, no scores.
    http://localhost:8765/carer     the carer / clinician view: timeline, levels, layer details, assessment.

The browser cannot run Python, so this script serves the pages and runs the orchestrator
(safety agent + companion LLM) behind the API. Standard library only. It listens on 127.0.0.1,
and with LM Studio nothing leaves this computer.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safety.agent import MODES, SafetyAgent, load_config  # noqa: E402
from safety.llm_providers import get_provider  # noqa: E402

from demo.companion import Companion, load_scenarios  # noqa: E402
from demo.orchestrator import Orchestrator  # noqa: E402

WEB = Path(__file__).parent / "web"
PAGES = {"/": "patient.html", "/patient": "patient.html", "/carer": "carer.html"}
PROVIDERS = ("lmstudio", "mock", "openai", "gemini")
ALLOWED_HOSTS = {"localhost", "127.0.0.1"}
MAX_SESSIONS = 50


def lmstudio_problem(provider) -> str | None:
    """None if LM Studio answers and lists the model, else what is wrong."""
    try:
        models = provider.list_models()
    except Exception:
        return (f"LM Studio is not reachable at {provider.base_url}. Open LM Studio, load the model, then "
                "Developer -> Start Server. Until then the companion cannot reply and unclear risky "
                "messages are treated as a crisis.")
    if not models:
        return "LM Studio is running but no model is loaded."
    if provider.model not in models:
        return f"LM Studio does not list the model '{provider.model}' (it has: {', '.join(models)}). Load it or change llm.model in config.yaml."
    return None


class Demo:
    """Agents, companions and open sessions, all in memory. One instance per server."""

    def __init__(self, cfg: dict, provider: str, mode: str):
        self.cfg = cfg
        self.provider, self.mode = provider, mode           # defaults for new sessions
        self.lock = threading.Lock()
        self.scenarios = load_scenarios()
        self.agents: dict[tuple[str, str], SafetyAgent] = {}
        self.companions: dict[str, Companion | None] = {}
        self.sessions: dict[str, tuple[Orchestrator, object]] = {}

    # ------------------------------------------------------------------ building blocks
    def agent(self, provider: str | None = None, mode: str | None = None) -> SafetyAgent:
        key = (provider or self.provider, mode or self.mode)
        if key[0] not in PROVIDERS or key[1] not in MODES:
            raise ValueError(f"unknown provider or mode: {key}")
        if key not in self.agents:
            self.agents[key] = SafetyAgent.from_config(ROOT / "config.yaml", provider=key[0], mode=key[1])
        return self.agents[key]

    def companion(self, provider: str | None = None) -> Companion | None:
        name = provider or self.provider
        if self.companions.get(name) is None:   # not built yet, or failed last time: try again
            c = dict(self.cfg.get("companion") or {})
            llm = dict(self.cfg.get("llm") or {})
            model = (c.get("model") or llm.get("model")) if name == llm.get("provider") else None
            try:
                prov = get_provider(name, model, timeout_s=c.get("timeout_s", 120), base_url=llm.get("base_url"),
                                    disable_thinking=c.get("disable_thinking", True))
                self.companions[name] = Companion(prov, self.scenarios, temperature=c.get("temperature", 0.7),
                                                  max_tokens=c.get("max_tokens", 160),
                                                  history_turns=c.get("history_turns", 12))
            except Exception as e:
                print(f"[demo] companion LLM unavailable ({e}); replies fall back to a fixed sentence")
                self.companions[name] = None
        return self.companions[name]

    def orchestrator(self, provider: str | None = None, mode: str | None = None) -> Orchestrator:
        return Orchestrator(self.agent(provider, mode), self.companion(provider), self.scenarios)

    def status(self) -> dict:
        agent, comp = self.agent(), self.companion()
        llm = agent.llm.provider if agent.llm else None
        out = {"mode": agent.mode, "provider": getattr(llm, "name", None), "model": getattr(llm, "model", None),
               "classifier": agent.clf.clf.name if agent.clf else None,
               "classifier_device": getattr(agent.clf.clf, "device", None) if agent.clf else None,
               "companion": comp.name if comp else None, "llm_problem": None,
               "scenarios": {k: {"title": v["title"], "blurb": v.get("blurb", "")} for k, v in self.scenarios.items()}}
        if llm is None:
            out["llm_problem"] = "No LLM is connected: the companion cannot reply and unclear risky messages are treated as a crisis."
        elif out["provider"] == "mock":
            out["llm_problem"] = "Offline test mode: canned companion replies and a rough stand-in for the LLM safety check."
        elif out["provider"] == "lmstudio":
            out["llm_problem"] = lmstudio_problem(llm)
        return out

    def session(self, sid: str | None):
        if sid not in self.sessions:
            raise LookupError("This conversation is gone (the server was restarted). Start a new one.")
        return self.sessions[sid]

    # ------------------------------------------------------------------ API routes: patient
    def api_session(self, b: dict) -> dict:
        orch = self.orchestrator()
        sess = orch.start(b.get("scenario", "free_chat"), b.get("age_group", "adult"), bool(b.get("carer_consent")))
        self.sessions[sess.id] = (orch, sess)
        while len(self.sessions) > MAX_SESSIONS:
            self.sessions.pop(next(iter(self.sessions)))
        first = sess.log[-1]
        return orch.patient_view(sess, first["text"], first["choices"])

    def api_message(self, b: dict) -> dict:
        orch, sess = self.session(b.get("session_id"))
        return orch.message(sess, b.get("text", ""))

    def api_event(self, b: dict) -> dict:
        orch, sess = self.session(b.get("session_id"))
        return orch.event(sess, b.get("label", ""))

    # ------------------------------------------------------------------ API routes: carer
    def api_sessions(self) -> list[dict]:
        return [orch.summary(sess) for orch, sess in reversed(list(self.sessions.values()))]

    def api_session_detail(self, sid: str | None) -> dict:
        orch, sess = self.session(sid)
        return orch.carer_view(sess)

    def api_reply_check(self, b: dict) -> dict:
        g = self.agent().guard.check(b.get("reply", ""), None)
        return {"ok": g.ok, "action": g.action, "text": g.text, "violations": g.violations,
                "warnings": g.warnings, "hint": g.regenerate_hint}

    def api_settings(self, b: dict) -> dict:
        if b.get("provider") in PROVIDERS:
            self.provider = b["provider"]
        if b.get("mode") in MODES:
            self.mode = b["mode"]
        return {"provider": self.provider, "mode": self.mode, "status": self.status()}


POST_ROUTES = {"/api/session": Demo.api_session, "/api/message": Demo.api_message, "/api/event": Demo.api_event,
               "/api/carer/reply_check": Demo.api_reply_check, "/api/carer/settings": Demo.api_settings}


def make_handler(demo: Demo):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def _send(self, code: int, body, ctype: str = "application/json; charset=utf-8") -> None:
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _local(self) -> bool:
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            return host in ALLOWED_HOSTS

        def do_GET(self):
            if not self._local():
                return self._send(403, {"error": "Open the page as http://localhost:<port>."})
            url = urlparse(self.path)
            q = parse_qs(url.query)
            try:
                if url.path in PAGES:
                    return self._send(200, (WEB / PAGES[url.path]).read_bytes(), "text/html; charset=utf-8")
                if url.path == "/api/status":
                    return self._send(200, demo.status())
                if url.path == "/api/carer/sessions":
                    return self._send(200, demo.api_sessions())
                if url.path == "/api/carer/session":
                    return self._send(200, demo.api_session_detail((q.get("id") or [None])[0]))
                if url.path == "/api/carer/settings":
                    return self._send(200, {"provider": demo.provider, "mode": demo.mode, "providers": PROVIDERS, "modes": MODES})
            except LookupError as e:
                return self._send(404, {"error": str(e)})
            except Exception as e:
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._local():
                return self._send(403, {"error": "Open the page as http://localhost:<port>."})
            route = POST_ROUTES.get(urlparse(self.path).path)
            if route is None:
                return self._send(404, {"error": "not found"})
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                return self._send(415, {"error": "JSON only"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                with demo.lock:  # one message at a time; the demo has one user
                    out = route(demo, body)
                self._send(200, out)
            except (LookupError, ValueError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # show the problem on the page instead of a silent failure
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def make_server(demo: Demo, port: int) -> ThreadingHTTPServer:
    for p in range(port, port + 10):  # next free port if this one is taken
        try:
            return ThreadingHTTPServer(("127.0.0.1", p), make_handler(demo))
        except OSError:
            continue
    raise OSError(f"No free port between {port} and {port + 9}")


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    ap = argparse.ArgumentParser(description="CompanionAI local demo (patient page + carer page)")
    ap.add_argument("--provider", choices=PROVIDERS, default=(cfg.get("llm") or {}).get("provider") or "mock")
    ap.add_argument("--mode", choices=MODES, default=cfg.get("mode", "full"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="don't open the browser")
    ap.add_argument("--pages", default="patient,carer", help="which pages to open (default both)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    demo = Demo(cfg, args.provider, args.mode)
    print("Loading the safety agent and the companion ...")
    st = demo.status()
    print(f"  safety LLM: {st['provider']}/{st['model']}   companion: {st['companion'] or 'none'}   "
          f"classifier: {st['classifier'] or 'not loaded'}"
          + (f" ({st['classifier_device']})" if st.get("classifier_device") else "") + f"   mode: {st['mode']}")
    if st["llm_problem"]:
        print(f"  Note: {st['llm_problem']}")

    server = make_server(demo, args.port)
    base = f"http://localhost:{server.server_address[1]}"
    print(f"\n  Patient page (the user):      {base}/patient\n  Carer page (carer / doctor):  {base}/carer")
    print("Leave this window open while you use the pages. Press Ctrl+C to stop.")
    if not args.no_browser:
        for i, page in enumerate(p.strip() for p in args.pages.split(",") if p.strip()):
            threading.Timer(0.6 + 1.2 * i, webbrowser.open, args=(f"{base}/{page}",), kwargs={"new": 1}).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
