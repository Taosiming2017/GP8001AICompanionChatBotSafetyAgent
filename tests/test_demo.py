"""The demo: orchestrator + the two-page server API, run with the offline mock LLM."""
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safety.agent import SafetyAgent, load_config  # noqa: E402
from safety.llm_providers import MockProvider  # noqa: E402

from demo import server  # noqa: E402
from demo.companion import Companion  # noqa: E402
from demo.orchestrator import Orchestrator  # noqa: E402

OPEN = urllib.request.build_opener(urllib.request.ProxyHandler({})).open  # never send localhost through a proxy
PATIENT_KEYS = {"reply", "choices", "paused", "ended", "session_id"}


def make_orch(mode="rules+llm"):
    agent = SafetyAgent.from_config(ROOT / "config.yaml", provider="mock", mode=mode)
    return Orchestrator(agent, Companion(MockProvider()))


# ------------------------------------------------------------------ orchestrator
def test_patient_view_never_carries_levels_or_scores():
    orch = make_orch()
    sess = orch.start("hawker_centre", age_group="teen")
    assert "Uncle Lim" in sess.log[0]["text"]
    out = orch.message(sess, "One chicken rice please.")
    assert set(out) == PATIENT_KEYS and out["reply"] and not out["paused"]
    text = json.dumps(out).lower()
    for word in ("level", "score", "classifier", "probs", "crisis", "layer"):
        assert word not in text, word


def test_crisis_flow_fixed_reply_then_support_then_release():
    orch = make_orch()
    sess = orch.start("free_chat", age_group="teen")
    out = orch.message(sess, "i want to die")
    assert "1767" in out["reply"] and "1771" in out["reply"] and out["paused"]
    assert "I am safe now. Go back to practice" in out["choices"]
    assert sess.log[-1]["source"] == "safety_fixed" and sess.log[-2]["decision"]["carer_alert"] is True

    out = orch.message(sess, "ok")                      # crisis hold: companion in support mode
    assert out["paused"] and "1767" in out["reply"]     # guard appends the support line if missing
    out = orch.event(sess, "I am safe now. Go back to practice")
    assert not sess.state.crisis_hold and "glad you are safe" in out["reply"]


def test_buttons_break_feelings_and_stop():
    orch = make_orch()
    sess = orch.start("job_interview")
    out = orch.event(sess, "Take a short break")
    assert "I am back" in out["choices"] and not out["ended"]
    out = orch.event(sess, "Worried")
    assert "Take a short break" in out["choices"]
    out = orch.event(sess, "Stop for today")
    assert out["ended"] and sess.ended
    with pytest.raises(ValueError):
        orch.message(sess, "hello again")


def test_companion_reply_is_guarded(monkeypatch):
    orch = make_orch()
    sess = orch.start("hawker_centre")
    said = iter(["Nice! That was 8/10. Do you want chilli?", "Do you want chilli with it?"])
    monkeypatch.setattr(orch.companion, "reply", lambda *a, **k: next(said))
    out = orch.message(sess, "One chicken rice.")
    assert "8/10" not in out["reply"]
    assert sess.log[-1]["source"] in ("llm", "llm_regenerated") and sess.log[-1]["guard"][0]["violations"]


def test_carer_view_has_everything():
    orch = make_orch()
    sess = orch.start("new_colleague", age_group="adult", carer_consent=True)
    orch.message(sess, "I am in the data team.")
    v = orch.carer_view(sess)
    user_rows = [r for r in v["log"] if r["kind"] == "user"]
    assert v["n_user_turns"] == 1 and user_rows[0]["decision"]["layers"][0]["layer"] == "rules"
    assert v["scenario_title"] and "session_level" in v


# ------------------------------------------------------------------ server API
@pytest.fixture(scope="module")
def base():
    demo = server.Demo(load_config(ROOT / "config.yaml"), "mock", "rules+llm")
    srv = server.make_server(demo, 0)  # port 0 = any free port
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://localhost:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(base, path):
    with OPEN(base + path, timeout=10) as r:
        return json.load(r)


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with OPEN(req, timeout=10) as r:
        return json.load(r)


def test_pages_and_status(base):
    with OPEN(base + "/patient", timeout=10) as r:
        assert b"<title>Companion</title>" in r.read()
    with OPEN(base + "/carer", timeout=10) as r:
        assert b"Carer view" in r.read()
    st = get(base, "/api/status")
    assert st["provider"] == "mock" and "hawker_centre" in st["scenarios"]


def test_patient_and_carer_endpoints(base):
    s = post(base, "/api/session", {"scenario": "hawker_centre", "age_group": "teen"})
    assert set(s) == PATIENT_KEYS and "order" in s["reply"].lower()
    sid = s["session_id"]
    out = post(base, "/api/message", {"session_id": sid, "text": "i want to die"})
    assert "1767" in out["reply"] and out["paused"]
    out = post(base, "/api/event", {"session_id": sid, "label": "I am safe now. Go back to practice"})
    assert not out["paused"] or out["reply"]
    lst = get(base, "/api/carer/sessions")
    assert lst and lst[0]["id"] == sid and lst[0]["crisis_count"] == 1
    detail = get(base, f"/api/carer/session?id={sid}")
    assert any(r["kind"] == "user" and r["decision"]["message_level"] == 3 for r in detail["log"])
    g = post(base, "/api/carer/reply_check", {"reply": "Good job! That was 8/10."})
    assert "8/10" not in g["text"]


def test_refuses_other_hosts_and_unknown_sessions(base):
    req = urllib.request.Request(base + "/api/status", headers={"Host": "evil.example"})
    with pytest.raises(urllib.error.HTTPError) as e:
        OPEN(req, timeout=10)
    assert e.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as e:
        post(base, "/api/message", {"session_id": "nope", "text": "hi"})
    assert e.value.code == 400
