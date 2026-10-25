"""SafetyAgent: the one object the chat app talks to.

    agent = SafetyAgent.from_config("config.yaml")
    session = agent.new_session(age_group="teen")
    decision = agent.check_user_message(session, text, context=last_companion_line, scenario="hawker_centre")
    ... if decision.fixed_reply: show it; else generate a reply with decision.companion_directives
    guard = agent.check_reply(session, decision, reply)

Layer order for each user message (proposal Section 4.1, plus the trained classifier):
    rules  ->  classifier  ->  LLM (only when needed)  ->  fusion  ->  session tracker  ->  policy
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import yaml

from .classifier import ClassifierLayer, load_classifier
from .llm_check import LLMCheckLayer
from .llm_providers import get_provider
from .output_guard import OutputGuard
from .policy import Policy
from .rules import RuleEngine
from .schema import Assessment, GuardResult, Level, SafetyDecision, Turn
from .state import SessionState, SessionTracker, StateConfig
from .text import word_count

MODES = ("rules", "classifier", "rules+classifier", "rules+llm", "full")


def load_config(path: Union[str, Path] = "config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


class SafetyAgent:
    def __init__(self, classifier=None, llm_provider=None, mode: str = "full",
                 state_cfg: Optional[StateConfig] = None, fail_safe: bool = True):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.rules = RuleEngine()
        self.clf = ClassifierLayer(classifier) if classifier is not None else None
        self.llm = LLMCheckLayer(llm_provider) if llm_provider is not None else None
        self.tracker = SessionTracker(state_cfg)
        self.policy = Policy()
        self.guard = OutputGuard()
        self.fail_safe = fail_safe

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, path: Union[str, Path] = "config.yaml", provider: Optional[str] = None,
                    mode: Optional[str] = None) -> "SafetyAgent":
        """Build from config.yaml. `provider` / `mode` override the file (e.g. provider="mock")."""
        cfg = load_config(path)
        llm_cfg = dict(cfg.get("llm") or {})
        if provider and provider != llm_cfg.get("provider"):
            llm_cfg["provider"] = provider
            llm_cfg["model"] = None  # the model in the file belongs to the other provider; use the default
        root = Path(path).resolve().parent

        clf_cfg = cfg.get("classifier") or {}
        clf = None
        for key in ("model_dir", "fallback_dir"):
            d = clf_cfg.get(key)
            if d and (root / d / "meta.json").exists():
                clf = load_classifier(root / d, device=clf_cfg.get("device"))
                break

        llm = None
        if llm_cfg.get("provider"):
            try:
                llm = get_provider(llm_cfg["provider"], llm_cfg.get("model"), llm_cfg.get("timeout_s", 20),
                                   llm_cfg.get("base_url"))
            except Exception as e:  # missing package or key: run without the LLM layer (fail-safe applies)
                print(f"[safety] LLM provider unavailable ({e}); continuing without it")
        return cls(classifier=clf, llm_provider=llm, mode=mode or cfg.get("mode", "full"),
                   state_cfg=StateConfig(**(cfg.get("state") or {})), fail_safe=cfg.get("fail_safe", True))

    def new_session(self, user_id: str = "demo", age_group: str = "adult",
                    carer_alert_consent: bool = False) -> SessionState:
        return SessionState(user_id=user_id, age_group=age_group, carer_alert_consent=carer_alert_consent)

    # ------------------------------------------------------------------ stateless message assessment
    def assess_message(self, text: str, context: Optional[str] = None, scenario: Optional[str] = None) -> Assessment:
        """Judge one message without touching any session. Used directly by evaluate.py."""
        use_rules = self.mode != "classifier"
        use_clf = self.mode in ("classifier", "rules+classifier", "full") and self.clf is not None
        use_llm = self.mode in ("rules+llm", "full") and self.llm is not None
        layers, notes = [], []

        # 1. rules: a high-precision match ends the assessment at once
        r = self.rules.assess(text)
        if use_rules:
            layers.append(r.to_layer())
            if r.is_decisive:
                return Assessment(Level.CRISIS, r.category, layers, False, ["decided by rules"])

        # 2. classifier
        c = self.clf.assess(text, context, scenario) if use_clf else None
        if c is not None:
            layers.append(c)
            # A confident "crisis" that rests on an idiom the rules already recognised
            # ("my phone died") is not trusted alone: the LLM gets the final word.
            only_idiom = use_rules and r.idioms and not (r.soft or r.negated)
            if c.level == Level.CRISIS and not c.wants_llm and only_idiom:
                c.wants_llm = c.risk_signal = True
                c.reasons.append("rules found only an idiom -> LLM must confirm")
            # Taxonomy rule: a short blunt reply ("no", "ok", "card") is not distress on its own.
            # With no cue words the classifier has nothing to go on, so it may not raise 1-2 word
            # messages above 0 by itself; a crisis guess on such a message goes to the LLM.
            if use_rules and word_count(text) <= 2 and not r.has_cues:
                if c.level in (Level.TENSE, Level.OVERWHELMED):
                    c.reasons.append(f"1-2 word reply without cue words: {c.level.name.lower()} -> safe (short is not distress)")
                    c.level = Level.SAFE
                elif c.level == Level.CRISIS and not c.wants_llm:
                    c.wants_llm = c.risk_signal = True
                    c.reasons.append("1-2 word reply without cue words -> LLM must confirm")

        if self.mode == "rules":
            return Assessment(r.rules_only_level(), None, layers, False, notes)
        if self.mode == "classifier":
            level = c.level if c else Level.SAFE
            return Assessment(level, "unspecified" if level == Level.CRISIS else None, layers, False, notes)

        # 3. LLM, only when a layer asks for it (or when there is no classifier to judge a cue)
        risk_signal = any(l.risk_signal for l in layers)
        need_llm = any(l.wants_llm for l in layers) or (c is None and use_rules and r.cue_level is not None)
        llm_called = False
        if need_llm and use_llm:
            llm_called = True
            v = self.llm.assess(text, context, scenario)
            layers.append(v)
            if v.error is None:
                return Assessment(v.level, v.category, layers, True, ["decided by LLM"])
            notes.append(f"LLM failed: {v.error}")
        if need_llm and risk_signal and self.fail_safe and self.mode in ("rules+llm", "full"):
            why = "without a working LLM" if llm_called else "but no LLM configured"
            notes.append(f"fail-safe: risk signal {why} -> crisis")
            return Assessment(Level.CRISIS, "unspecified", layers, llm_called, notes)

        # 4. fusion without the LLM
        if c is not None:
            level = c.level
            if use_rules and r.cue_level == Level.OVERWHELMED and level < Level.OVERWHELMED:
                level = Level.OVERWHELMED   # explicit overload words ("too loud", "please stop") win
                notes.append("rules: overload words raised the level to 2")
        elif use_rules:
            level = r.rules_only_level()
        else:
            level = Level.SAFE
        return Assessment(level, "unspecified" if level == Level.CRISIS else None, layers, llm_called, notes)

    # ------------------------------------------------------------------ session-level API
    def check_user_message(self, s: SessionState, text: str,
                           context: Union[None, str, list[Turn]] = None,
                           scenario: Optional[str] = None) -> SafetyDecision:
        if isinstance(context, list):  # take the last companion line from the history
            context = next((t.text for t in reversed(context) if t.role == "companion"), None)
        a = self.assess_message(text, context, scenario)
        prev = s.level
        _, state_notes = self.tracker.update(s, a.level, word_count(text))
        d = self.policy.decide(s, a.level, prev, a.category, a.layers, a.llm_called, a.notes + state_notes)
        s.history.append((int(a.level), int(s.level), d.action))
        return d

    def check_reply(self, s: SessionState, decision: SafetyDecision, reply: str) -> GuardResult:
        return self.guard.check(reply, decision)

    def release_crisis_hold(self, s: SessionState) -> None:
        self.tracker.release_crisis_hold(s)

    def user_chose_continue(self, s: SessionState) -> None:
        self.tracker.user_chose_continue(s)
