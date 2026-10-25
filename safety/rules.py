"""Layer 1: keyword / regex rules.

Fast, transparent, and the only layer that can send a message straight to Level 3
on its own. It also produces softer signals:
  * soft risk cues      -> the agent asks the LLM for a second opinion
  * tension / overwhelm -> used by the session tracker, and as the level
                           when no classifier is loaded (rules-only mode)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from . import lexicons as lx
from .schema import CRISIS_CATEGORIES, LayerResult, Level
from .text import normalize_for_rules

# Most severe first. Used only to pick the template and the logged category.
_CATEGORY_PRIORITY = [
    "cssrs_6_behaviour", "cssrs_5_plan", "cssrs_4_intent", "cssrs_3_method",
    "cssrs_2_active_thoughts", "nssi", "harm_others", "unsafe_abuse",
    "cssrs_1_wish_dead", "others_at_risk",
]
assert set(_CATEGORY_PRIORITY) == set(CRISIS_CATEGORIES)


def _compile(patterns):
    return [re.compile(p) for p in patterns]


@dataclass
class RuleOutput:
    normalized: str
    residual: str                                   # text after idioms were removed
    idioms: list[str] = field(default_factory=list)
    decisive: list[tuple[str, str]] = field(default_factory=list)   # (category, matched text)
    negated: list[tuple[str, str]] = field(default_factory=list)
    soft: list[str] = field(default_factory=list)
    tension: list[str] = field(default_factory=list)
    overwhelm: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def is_decisive(self) -> bool:
        return bool(self.decisive)

    @property
    def category(self) -> Optional[str]:
        if not self.decisive:
            return None
        cats = {c for c, _ in self.decisive}
        return next(c for c in _CATEGORY_PRIORITY if c in cats)

    @property
    def wants_llm(self) -> bool:
        return not self.decisive and bool(self.soft or self.negated)

    @property
    def has_cues(self) -> bool:
        """Any risk, tension or overload words at all (idioms do not count)."""
        return bool(self.decisive or self.soft or self.negated or self.tension or self.overwhelm)

    @property
    def cue_level(self) -> Optional[Level]:
        """Level suggested by explicit words, ignoring decisive matches."""
        if self.overwhelm:
            return Level.OVERWHELMED
        if self.tension:
            return Level.TENSE
        return None

    def rules_only_level(self) -> Level:
        """Level used when no classifier and no LLM are available.

        Soft risk cues fail closed (treated as crisis) because nothing else can check them.
        """
        if self.decisive or self.soft or self.negated:
            return Level.CRISIS
        return self.cue_level or Level.SAFE

    def to_layer(self) -> LayerResult:
        reasons = []
        reasons += [f"decisive:{c}:'{m}'" for c, m in self.decisive]
        reasons += [f"negated:{c}:'{m}'" for c, m in self.negated]
        reasons += [f"soft:'{m}'" for m in self.soft]
        reasons += [f"overwhelm:'{m}'" for m in self.overwhelm]
        reasons += [f"tension:'{m}'" for m in self.tension]
        reasons += [f"idiom_removed:'{m}'" for m in self.idioms]
        return LayerResult(
            layer="rules",
            level=Level.CRISIS if self.decisive else self.cue_level,
            confidence=1.0 if self.decisive else (0.5 if self.cue_level else 0.0),
            category=self.category,
            reasons=reasons,
            decisive=self.is_decisive,
            wants_llm=self.wants_llm,
            risk_signal=self.wants_llm,   # soft cues and negated matches are possible crises
            latency_ms=self.latency_ms,
        )


class RuleEngine:
    def __init__(self):
        self.idioms = _compile(lx.IDIOMS)
        self.decisive = {cat: _compile(ps) for cat, ps in lx.DECISIVE.items()}
        self.soft = _compile(lx.SOFT_RISK)
        self.tension = _compile(lx.TENSION)
        self.overwhelm = _compile(lx.OVERWHELM)
        self.negation = re.compile(lx.NEGATION_BEFORE)

    def assess(self, text: str) -> RuleOutput:
        t0 = time.perf_counter()
        out = self._assess_norm(normalize_for_rules(text, squeeze=2))
        alt = normalize_for_rules(text, squeeze=1)   # catches stretched words: "dieee" -> "die"
        if alt != out.normalized:
            o1 = self._assess_norm(alt)
            for name in ("idioms", "decisive", "negated", "soft", "tension", "overwhelm"):
                merged = getattr(out, name)
                merged += [x for x in getattr(o1, name) if x not in merged]
        out.latency_ms = (time.perf_counter() - t0) * 1000
        return out

    def _assess_norm(self, norm: str) -> RuleOutput:
        out = RuleOutput(normalized=norm, residual=norm)
        residual = norm
        for pat in self.idioms:
            for m in pat.finditer(residual):
                out.idioms.append(m.group(0))
            residual = pat.sub(" ", residual)
        residual = re.sub(r"\s+", " ", residual).strip()
        out.residual = residual

        for cat, pats in self.decisive.items():
            for pat in pats:
                for m in pat.finditer(residual):
                    prefix = residual[max(0, m.start() - 40): m.start()]
                    if self.negation.search(prefix):
                        out.negated.append((cat, m.group(0)))
                    else:
                        out.decisive.append((cat, m.group(0)))

        out.soft = [m.group(0) for p in self.soft for m in p.finditer(residual)]
        out.tension = [m.group(0) for p in self.tension for m in p.finditer(residual)]
        out.overwhelm = [m.group(0) for p in self.overwhelm for m in p.finditer(residual)]
        return out
