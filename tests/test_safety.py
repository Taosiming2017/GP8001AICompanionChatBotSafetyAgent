"""Unit tests: python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety.agent import SafetyAgent  # noqa: E402
from safety.classifier import DEFAULT_THRESHOLDS  # noqa: E402
from safety.llm_providers import MockProvider  # noqa: E402
from safety.output_guard import OutputGuard  # noqa: E402
from safety.rules import RuleEngine  # noqa: E402
from safety.schema import Action, Level  # noqa: E402
from safety.state import SessionState, SessionTracker  # noqa: E402

R = RuleEngine()


# ------------------------------------------------------------------ rules
@pytest.mark.parametrize("text,cat", [
    ("i want to die", "cssrs_1_wish_dead"),
    ("Everyone would be better off without me.", "cssrs_1_wish_dead"),
    ("I've been thinking about killing myself", "cssrs_2_active_thoughts"),
    ("kms", "cssrs_2_active_thoughts"),
    ("i want to unalive myself", "cssrs_2_active_thoughts"),
    ("I'm going to end my life", "cssrs_4_intent"),
    ("I tried to kill myself last year", "cssrs_6_behaviour"),
    ("i have been cutting myself again", "nssi"),
    ("my friend said she wants to die", "others_at_risk"),
    ("i don't feel safe at home", "unsafe_abuse"),
    ("im not okay. i want to kill myself", "cssrs_2_active_thoughts"),
])
def test_decisive(text, cat):
    out = R.assess(text)
    assert out.is_decisive and out.category == cat, (text, out.decisive)


@pytest.mark.parametrize("text", [
    "I'd kill for some chicken rice", "my phone died", "this chilli is to die for", "the deadline is killing me",
    "I'm dying to try the laksa", "I was bored to death", "I want to die my hair blue", "No.", "card",
])
def test_idioms_and_blunt_are_clean(text):
    out = R.assess(text)
    assert not out.is_decisive and not out.soft, (text, out)


def test_negation_goes_to_llm_not_decisive():
    out = R.assess("I don't want to kill myself, I'm just tired")
    assert not out.is_decisive and out.wants_llm


def test_soft_cue_asks_llm():
    assert R.assess("I wish I could disappear").wants_llm


def test_zero_width_and_stretched_letters():
    assert R.assess("i want to d​ie").is_decisive
    assert R.assess("i wanttt to dieee").is_decisive


# ------------------------------------------------------------------ state tracker
def test_terse_user_not_flagged():
    tr, s = SessionTracker(), SessionState()
    for w in [2, 1, 3, 1, 1, 2, 1]:
        lvl, _ = tr.update(s, Level.SAFE, w)
    assert lvl == Level.SAFE


def test_length_drop_flags_tense_then_recovers():
    tr, s = SessionTracker(), SessionState()
    for w in [15, 17, 15, 17]:
        tr.update(s, Level.SAFE, w)
    assert tr.update(s, Level.SAFE, 1)[0] == Level.SAFE      # one short reply is fine
    assert tr.update(s, Level.SAFE, 1)[0] == Level.TENSE     # 2 of last 3 short
    assert tr.update(s, Level.SAFE, 15)[0] == Level.TENSE    # hysteresis
    assert tr.update(s, Level.SAFE, 15)[0] == Level.SAFE


def test_crisis_hold_needs_explicit_release():
    tr, s = SessionTracker(), SessionState()
    tr.update(s, Level.CRISIS, 5)
    for _ in range(5):
        assert tr.update(s, Level.SAFE, 5)[0] == Level.CRISIS
    tr.release_crisis_hold(s)
    assert s.level == Level.TENSE and not s.crisis_hold


# ------------------------------------------------------------------ output guard
G = OutputGuard()


def test_score_sentences_are_stripped():
    r = G.check("Nice try! I would give that 7/10. Can you say it again a bit slower?")
    assert "7/10" not in r.text and "slower" in r.text and r.violations


def test_hard_violations_block():
    for bad in ["I am a real human, trust me.", "You probably have anxiety.", "You should take your medication.",
                "I’m a real therapist, you can trust me."]:   # curly apostrophe
        assert not G.check(bad).ok, bad


def test_percent_scores_are_stripped():
    r = G.check("Well done. That was 80% correct. Try once more.")
    assert "80%" not in r.text and "Try once more" in r.text and r.violations


def test_not_a_human_is_fine():
    assert G.check("I am not a human. I am an AI.").ok


def test_idiom_reply_triggers_regenerate():
    r = G.check("Don't worry, it will be a piece of cake.")
    assert r.action == "regenerate" and "literal" in r.regenerate_hint


# ------------------------------------------------------------------ agent fusion
class FakeClf:
    """Classifier stub returning fixed probabilities."""

    def __init__(self, p):
        self.p = np.array([p])
        self.thresholds = dict(DEFAULT_THRESHOLDS)

    def predict_proba(self, texts, contexts, temperature=None):
        return self.p


def test_rules_decisive_skips_llm():
    llm = MockProvider(0)
    a = SafetyAgent(classifier=FakeClf([0.9, 0.05, 0.03, 0.02]), llm_provider=llm)
    r = a.assess_message("i want to die")
    assert r.level == Level.CRISIS and not r.llm_called and llm.calls == 0


def test_grey_zone_uses_llm_verdict():
    a = SafetyAgent(classifier=FakeClf([0.6, 0.1, 0.1, 0.2]), llm_provider=MockProvider(0))
    r = a.assess_message("I think it is all pointless")
    assert r.llm_called and r.level == Level.SAFE


def test_llm_failure_fails_safe():
    a = SafetyAgent(classifier=FakeClf([0.6, 0.1, 0.1, 0.2]), llm_provider=MockProvider("fail"))
    r = a.assess_message("I think it is all pointless")
    assert r.level == Level.CRISIS and any("fail-safe" in n for n in r.notes)


def test_unsure_level2_with_llm_down_is_not_a_crisis():
    """Classifier unsure between tense and overwhelmed, LLM unreachable: no risk signal, so no fail-safe."""
    a = SafetyAgent(classifier=FakeClf([0.1, 0.4, 0.45, 0.05]), llm_provider=MockProvider("fail"))
    r = a.assess_message("I think I need a minute")
    assert r.level == Level.OVERWHELMED and not any("fail-safe" in n for n in r.notes)


def test_idiom_plus_confident_classifier_crisis_goes_to_llm():
    """'my phone died' with an over-confident classifier: the rules saw only an idiom, so the LLM decides."""
    llm = MockProvider(0)
    a = SafetyAgent(classifier=FakeClf([0.02, 0.02, 0.01, 0.95]), llm_provider=llm)
    r = a.assess_message("my phone died this morning")
    assert r.llm_called and r.level == Level.SAFE and llm.calls == 1


def test_short_blunt_reply_is_not_distress_even_if_classifier_says_so():
    a = SafetyAgent(classifier=FakeClf([0.2, 0.1, 0.7, 0.0]), llm_provider=MockProvider(0))
    assert a.assess_message("no").level == Level.SAFE
    assert a.assess_message("no, it is too loud").level == Level.OVERWHELMED   # cue words: classifier stands
    llm = MockProvider(0)
    a = SafetyAgent(classifier=FakeClf([0.02, 0.02, 0.01, 0.95]), llm_provider=llm)
    r = a.assess_message("bye")
    assert r.llm_called and r.level == Level.SAFE                                # crisis guess on "bye": LLM decides


def test_overload_words_raise_a_confident_safe():
    a = SafetyAgent(classifier=FakeClf([0.9, 0.05, 0.03, 0.02]), llm_provider=MockProvider(0))
    r = a.assess_message("too loud, please stop")
    assert r.level == Level.OVERWHELMED and not r.llm_called


def test_crisis_decision_has_fixed_helplines_and_teen_alert():
    a = SafetyAgent(classifier=FakeClf([0.9, 0.05, 0.03, 0.02]), llm_provider=MockProvider(0))
    s = a.new_session(age_group="teen")
    d = a.check_user_message(s, "i want to die", scenario="free_chat")
    assert d.action == Action.CRISIS_PAUSE and "1767" in d.fixed_reply and "1771" in d.fixed_reply
    assert d.carer_alert and d.pause_roleplay and d.suppress_feedback
    d2 = a.check_user_message(s, "ok", scenario="free_chat")
    assert d2.action == Action.CRISIS_SUPPORT
    assert "1767" in a.check_reply(s, d2, "I am here with you.").text


# ------------------------------------------------------------------ LM Studio provider (fake local server)
def test_lmstudio_provider_against_fake_server():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from safety.llm_check import LLMCheckLayer
    from safety.llm_providers import get_provider

    seen = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # /v1/models
            self._send({"object": "list", "data": [{"id": "local-test-model", "object": "model", "owned_by": "me"}]})

        def do_POST(self):  # /v1/chat/completions
            seen["req"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            content = json.dumps({"level": 3, "category": "cssrs_1_wish_dead", "confidence": 0.9, "rationale": "test"})
            self._send({"id": "x", "object": "chat.completion", "created": 0, "model": "local-test-model",
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": content}}]})

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        prov = get_provider("lmstudio", base_url=f"http://127.0.0.1:{srv.server_port}/v1")
        assert prov.model == "local-test-model"          # picked the first loaded model
        res = LLMCheckLayer(prov).assess("nobody would notice if i was gone", None, "free_chat")
        assert res.level == Level.CRISIS and res.category == "cssrs_1_wish_dead"
        rf = seen["req"]["response_format"]
        assert rf["type"] == "json_schema" and rf["json_schema"]["schema"]["required"][0] == "level"
        assert seen["req"]["chat_template_kwargs"] == {"enable_thinking": False}   # Qwen3 thinking switched off
        assert prov.chat([{"role": "user", "content": "hi"}]).startswith("{")      # chat() returns plain text
    finally:
        srv.shutdown()


def test_provider_override_drops_the_files_model(tmp_path, monkeypatch):
    """config.yaml names a local LM Studio model; switching to openai must not send that name to OpenAI."""
    import safety.agent as ag

    seen = {}

    def fake_get_provider(name, model=None, *args):
        seen.update(name=name, model=model)
        return MockProvider()

    monkeypatch.setattr(ag, "get_provider", fake_get_provider)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("mode: full\nllm: {provider: lmstudio, model: some-local-model}\n", encoding="utf-8")
    SafetyAgent.from_config(cfg, provider="openai")
    assert seen == {"name": "openai", "model": None}
    SafetyAgent.from_config(cfg, provider="lmstudio")
    assert seen == {"name": "lmstudio", "model": "some-local-model"}
