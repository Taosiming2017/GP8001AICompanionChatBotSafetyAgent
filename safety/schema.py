"""Shared types for the CompanionAI safety agent."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class Level(IntEnum):
    """Graded safety levels (proposal Section 4.1)."""

    SAFE = 0         # ordinary message, including blunt or literal ones
    TENSE = 1        # signs of tension -> slow down, shorter sentences, two choices
    OVERWHELMED = 2  # overwhelmed -> pause role-play, offer one playbook prompt
    CRISIS = 3       # possible crisis -> pause, show Singapore helplines


LABELS = ["safe", "tense", "overwhelmed", "crisis"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}

# Crisis sub-categories. cssrs_1..cssrs_6 follow the six screener items of the
# Columbia Suicide Severity Rating Scale (C-SSRS) [23].
CRISIS_CATEGORIES = [
    "cssrs_1_wish_dead",
    "cssrs_2_active_thoughts",
    "cssrs_3_method",
    "cssrs_4_intent",
    "cssrs_5_plan",
    "cssrs_6_behaviour",
    "nssi",             # non-suicidal self-harm
    "harm_others",
    "others_at_risk",   # user is worried about someone else
    "unsafe_abuse",     # user is being hurt or is unsafe where they are
]


class Action:
    CONTINUE = "continue"
    SLOW_DOWN = "slow_down"
    PAUSE_REGULATE = "pause_regulate"
    CRISIS_PAUSE = "crisis_pause"      # new crisis message: fixed helpline reply
    CRISIS_SUPPORT = "crisis_support"  # later messages while the crisis hold is on


ACTION_FOR_LEVEL = {
    Level.SAFE: Action.CONTINUE,
    Level.TENSE: Action.SLOW_DOWN,
    Level.OVERWHELMED: Action.PAUSE_REGULATE,
    Level.CRISIS: Action.CRISIS_PAUSE,
}


@dataclass
class Turn:
    role: str  # "user" or "companion"
    text: str


@dataclass
class LayerResult:
    """What one detection layer concluded about one user message."""

    layer: str                      # "rules" | "classifier" | "llm"
    level: Optional[Level]          # None means "no opinion"
    confidence: float = 0.0
    category: Optional[str] = None  # e.g. "cssrs_2_active_thoughts"
    reasons: list[str] = field(default_factory=list)
    probs: Optional[dict[str, float]] = None
    decisive: bool = False          # rules only: a high-precision crisis match
    wants_llm: bool = False         # this layer asks for an LLM second opinion
    risk_signal: bool = False       # the reason for asking is a possible crisis (fail-safe applies)
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class Assessment:
    """What the detection layers concluded about ONE message, before any session state."""

    level: Level
    category: Optional[str]
    layers: list[LayerResult]
    llm_called: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class SafetyDecision:
    """Returned to the orchestrator for every user message."""

    message_level: Level          # this message on its own
    session_level: Level          # after the session state tracker
    action: str                   # one of Action.*
    category: Optional[str]
    layers: list[LayerResult]
    llm_called: bool
    fixed_reply: Optional[str] = None            # if set, show this instead of an LLM reply
    choices: list[str] = field(default_factory=list)  # buttons to show with fixed_reply
    companion_directives: list[str] = field(default_factory=list)
    suppress_feedback: bool = False
    pause_roleplay: bool = False
    carer_alert: bool = False     # would alert a carer (logged only in the demo)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "message_level": int(self.message_level),
            "session_level": int(self.session_level),
            "action": self.action,
            "category": self.category,
            "llm_called": self.llm_called,
            "fixed_reply": self.fixed_reply,
            "choices": self.choices,
            "companion_directives": self.companion_directives,
            "suppress_feedback": self.suppress_feedback,
            "pause_roleplay": self.pause_roleplay,
            "carer_alert": self.carer_alert,
            "notes": self.notes,
            "layers": [
                {
                    "layer": l.layer,
                    "level": None if l.level is None else int(l.level),
                    "confidence": round(l.confidence, 3),
                    "category": l.category,
                    "reasons": l.reasons,
                    "probs": l.probs,
                    "decisive": l.decisive,
                    "wants_llm": l.wants_llm,
                    "risk_signal": l.risk_signal,
                    "error": l.error,
                    "latency_ms": round(l.latency_ms, 1),
                }
                for l in self.layers
            ],
        }


@dataclass
class GuardResult:
    """Result of checking a companion reply before it is shown."""

    ok: bool                     # safe to show as is (after any automatic fixes)
    text: str                    # the text to show (may have sentences stripped)
    action: str                  # "show" | "show_fixed" | "regenerate" | "replace"
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    regenerate_hint: Optional[str] = None
