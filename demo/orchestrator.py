"""Orchestrator: one user turn = safety check -> companion reply -> output guard, plus the buttons.

This is the real version of the integration sketch in the proposal (Figure 1). The chat app
calls `message()` for typed text and `event()` for button presses, and shows what comes back.
The patient never sees levels or scores: `message()` returns only text, buttons and flags.
Everything else (layer results, levels, guard verdicts, timings) goes into `Session.log`,
which the carer page reads through `carer_view()`.

    orch = Orchestrator(agent, companion)
    sess = orch.start("hawker_centre", age_group="teen")
    out = orch.message(sess, "One chicken rice please")    # {"reply", "choices", "paused", "ended"}
    out = orch.event(sess, "Take a short break")
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from safety.agent import SafetyAgent
from safety.output_guard import SAFE_FALLBACK
from safety.schema import SafetyDecision
from safety.state import SessionState

from .companion import Companion


# Shown when the companion LLM cannot be reached at all (distinct from SAFE_FALLBACK, which is for a
# reply the output guard rejected twice). Plain words, no technical detail; the carer page has the error.
COMPANION_UNAVAILABLE = "I cannot answer right now. Please wait a moment and try again, or press a button."


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class Session:
    id: str
    scenario: str
    state: SessionState
    agent: SafetyAgent
    companion: Optional[Companion]
    started: str = field(default_factory=_now)
    log: list[dict] = field(default_factory=list)          # everything, for the carer view
    history: list[dict] = field(default_factory=list)      # [{"role","text"}], for the companion prompt
    last_decision: Optional[SafetyDecision] = None
    ended: bool = False

    @property
    def last_companion_line(self) -> Optional[str]:
        return next((h["text"] for h in reversed(self.history) if h["role"] == "companion"), None)

    def add(self, kind: str, **entry) -> dict:
        row = {"i": len(self.log), "t": _now(), "kind": kind, **entry}
        self.log.append(row)
        return row


class Orchestrator:
    def __init__(self, agent: SafetyAgent, companion: Optional[Companion], scenarios: Optional[dict] = None):
        self.agent, self.companion = agent, companion
        self.scenarios = scenarios or (companion.scenarios if companion else {})
        self.play = agent.policy.play
        self.support_line = agent.policy.support_line
        self.events = self.play.get("events", {})

    # ------------------------------------------------------------------ session lifecycle
    def start(self, scenario: str, age_group: str = "adult", carer_consent: bool = False) -> Session:
        if scenario not in self.scenarios:
            raise ValueError(f"unknown scenario {scenario!r}")
        s = self.agent.new_session(user_id=uuid.uuid4().hex[:8], age_group=age_group, carer_alert_consent=carer_consent)
        sess = Session(id=uuid.uuid4().hex, scenario=scenario, state=s, agent=self.agent, companion=self.companion)
        opener = self.scenarios[scenario]["opener"]
        self._say(sess, opener, source="scenario_opener", choices=[])
        return sess

    def patient_view(self, sess: Session, reply: str, choices: list[str]) -> dict:
        """What the patient page gets: no levels, no scores, no layer details."""
        return {"reply": reply, "choices": list(choices), "paused": sess.state.level >= 2 or sess.state.crisis_hold,
                "ended": sess.ended, "session_id": sess.id}

    # ------------------------------------------------------------------ one typed message
    def message(self, sess: Session, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            raise ValueError("Empty message.")
        if sess.ended:
            raise ValueError("This conversation has ended. Start a new one.")
        s = sess.state
        t0 = time.perf_counter()
        d = self.agent.check_user_message(s, text, context=sess.last_companion_line, scenario=sess.scenario)
        sess.last_decision = d
        sess.history.append({"role": "user", "text": text})
        sess.add("user", text=text, decision=d.to_dict(), session_level=int(s.level), crisis_hold=s.crisis_hold,
                 ms=round((time.perf_counter() - t0) * 1000))

        if d.fixed_reply:   # helplines / calming prompt: shown exactly as written, never LLM text
            self._say(sess, d.fixed_reply, source="safety_fixed", choices=d.choices, action=d.action)
            print(f"[demo] level {int(d.message_level)}/{int(s.level)} {d.action:<15} fixed reply"
                  + (f"  ({'; '.join(n for n in d.notes if 'fail-safe' in n or 'LLM failed' in n)})" if any('LLM' in n or 'fail-safe' in n for n in d.notes) else "")
                  + f"  | {text[:50]}")
            return self.patient_view(sess, d.fixed_reply, d.choices)

        reply, info = self._companion_reply(sess, d)
        self._say(sess, reply, choices=d.choices, action=d.action, **info)
        print(f"[demo] level {int(d.message_level)}/{int(s.level)} {d.action:<15} companion={info.get('source')} "
              f"{info.get('ms', 0)} ms  | {text[:50]}")
        return self.patient_view(sess, reply, d.choices)

    def _companion_reply(self, sess: Session, d: SafetyDecision) -> tuple[str, dict]:
        """Ask the companion, run the output guard, retry once with a hint, else fall back."""
        if self.companion is None:
            print("[demo] companion: no LLM configured -> fixed 'cannot answer' reply")
            return COMPANION_UNAVAILABLE, {"source": "unavailable", "guard": [], "note": "no companion LLM configured"}
        attempts, hint, t0 = [], None, time.perf_counter()
        for n in range(2):
            try:
                raw = self.companion.reply(sess.scenario, sess.history, d.companion_directives, d.pause_roleplay, hint)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"[:300]
                attempts.append({"error": err})
                print(f"[demo] companion LLM call failed ({self.companion.name}): {err}\n"
                      "       Is LM Studio running with the model loaded and the server on? Try: python scripts/check_llm.py")
                return COMPANION_UNAVAILABLE, {"source": "unavailable", "guard": attempts, "ms": _ms(t0)}
            g = self.agent.check_reply(sess.state, d, raw)
            attempts.append({"raw": raw, "action": g.action, "ok": g.ok, "violations": g.violations,
                             "warnings": g.warnings, "hint": g.regenerate_hint})
            if g.action in ("show", "show_fixed"):
                return g.text, {"source": "llm" if n == 0 else "llm_regenerated", "guard": attempts, "ms": _ms(t0)}
            hint = g.regenerate_hint
            last = g
        # second attempt still flagged: show it if the only problems are soft (length, idioms), else fall back
        if last.ok:
            return last.text, {"source": "llm_regenerated", "guard": attempts, "ms": _ms(t0)}
        return SAFE_FALLBACK, {"source": "fallback", "guard": attempts, "ms": _ms(t0)}

    # ------------------------------------------------------------------ buttons
    def event(self, sess: Session, label: str) -> dict:
        s = sess.state
        spec = self.events.get(label)
        feelings = self.events.get("feelings", {})
        if spec is None and label in feelings.get("labels", []):
            spec = feelings
        if spec is None:
            sess.add("event", label=label, effect="ignored (unknown button)")
            return self.patient_view(sess, "", [])

        effect = []
        if spec.get("call"):
            getattr(self.agent, spec["call"])(s)
            effect.append(spec["call"])
        if spec.get("end"):
            sess.ended = True
            effect.append("session ended")
        if spec.get("next_prompt"):
            p = self.agent.policy.playbook_prompt(s)
            text, choices = p["text"], list(p["choices"])
            effect.append(f"playbook prompt: {p['id']}")
        else:
            text, choices = spec["text"], list(spec.get("choices", []))
        if s.crisis_hold and not spec.get("call") == "release_crisis_hold":
            if self.support_line not in text:
                text = f"{text}\n{self.support_line}"
            if not sess.ended and not choices:
                choices = list(self.agent.policy.help["choices"])
        sess.add("event", label=label, effect=", ".join(effect) or "noted", session_level=int(s.level),
                 crisis_hold=s.crisis_hold)
        self._say(sess, text, source="button_reply", choices=choices)
        return self.patient_view(sess, text, choices)

    # ------------------------------------------------------------------ helpers
    def _say(self, sess: Session, text: str, source: str, choices: list[str], **extra) -> None:
        sess.history.append({"role": "companion", "text": text})
        sess.add("companion", text=text, source=source, choices=list(choices), **extra)

    def carer_view(self, sess: Session) -> dict:
        s = sess.state
        return {
            "id": sess.id, "scenario": sess.scenario, "scenario_title": self.scenarios[sess.scenario]["title"],
            "started": sess.started, "ended": sess.ended, "age_group": s.age_group,
            "carer_alert_consent": s.carer_alert_consent, "session_level": int(s.level),
            "crisis_hold": s.crisis_hold, "crisis_count": s.crisis_count,
            "n_user_turns": sum(1 for r in sess.log if r["kind"] == "user"),
            "carer_alerts": sum(1 for r in sess.log if r["kind"] == "user" and r["decision"]["carer_alert"]),
            "log": sess.log,
        }

    def summary(self, sess: Session) -> dict:
        v = self.carer_view(sess)
        return {k: v[k] for k in v if k != "log"}


def _ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)
