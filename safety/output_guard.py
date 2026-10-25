"""Checks every companion reply before it is shown (the second half of Figure 1).

Hard violations (must not reach the user):
  * score_leak    - scores, grades, right/wrong marks (proposal: never shown to the user)
  * harm_content  - wording about methods of self-harm or suicide
  * medical       - diagnosis, medication or treatment advice (CompanionAI is non-clinical)
  * impersonation - claiming to be a human or a professional
Soft warnings (regenerate once, then show):
  * too_long, long_sentences, idioms, sarcasm
"""
from __future__ import annotations

import re
from typing import Optional

from .schema import Action, GuardResult, SafetyDecision
from .text import normalize_quotes

SCORE = [
    r"\b\d+(\.\d+)?\s*/\s*(10|5|100)\b",
    r"\b\d{1,3}\s*(%|percent\b)",       # "80%", "80 percent" (no \b after %, it is not a word char)
    r"\b(score|scored|scoring|grade|graded|rating)\b",
    r"\b\d+ (marks|points)\b",
    r"\bincorrect\b",
    r"\b(wrong|right|correct) (answer|reply|response)\b",
    r"\b(that is|that's|thats|that was|this is) (wrong|incorrect)\b",
    r"\b(your|the|that) (answer|reply|response) (is|was) (wrong|right|correct|good|bad|perfect)\b",
    r"\byou (passed|failed|got it right|got it wrong)\b",
    r"[\u2605\u2606]",  # stars
]
HARM = [
    r"\b(overdose|lethal|noose|hang (yourself|myself)|slit)\b",
    r"\b(how|ways?) to (kill|hurt|harm|cut) (yourself|myself)\b",
    r"\b(ways?|how) to (die|end your life|commit suicide)\b",
]
MEDICAL = [
    r"\byou (have|might have|probably have|may have|seem to have) (autism|adhd|depression|anxiety|ocd|bipolar|a (mental )?disorder|an anxiety disorder)\b",
    r"\bdiagnos(e|is|ed|ing)\b",
    r"\b(prescri(be|ption)|medication|dosage|antidepressant)s?\b",
    r"\byou should (take|stop taking)\b",
]
IMPERSONATION = [
    r"\bi(m|'m| am) (?!not\b)(a |an )?(real |actual )?(human|person|therapist|doctor|counsell?or|psychologist|psychiatrist)\b",
]
IDIOMS = [
    "break the ice", "piece of cake", "hang in there", "hit the ground running", "under the weather",
    "spill the beans", "cost an arm and a leg", "beat around the bush", "on the same page",
    "take it with a grain of salt", "the ball is in your court", "bite the bullet", "cut corners",
    "a blessing in disguise", "speak of the devil", "once in a blue moon", "out of the blue",
    "hit the nail on the head", "let the cat out of the bag", "pull your leg", "kill two birds",
    "cold feet", "break a leg", "in hot water", "go the extra mile", "touch base", "circle back",
    "no-brainer", "rain check", "keep your chin up", "walk in the park",
]
SARCASM = [r"\byeah,? right\b", r"/s\b", r"\boh,? great\b", r"\bwhat a surprise\b"]

SAFE_FALLBACK = "Sorry, I made a mistake. Let's try that again. What would you like to say next?"


def _rx(ps):
    return [re.compile(p, re.IGNORECASE) for p in ps]


_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


class OutputGuard:
    def __init__(self, max_words: Optional[dict] = None, max_avg_sentence_words: float = 15.0):
        self.score, self.harm, self.medical = _rx(SCORE), _rx(HARM), _rx(MEDICAL)
        self.imp, self.sarcasm = _rx(IMPERSONATION), _rx(SARCASM)
        self.max_words = max_words or {"continue": 70, "slow_down": 25, "crisis_support": 45, "pause_regulate": 35}
        self.max_avg = max_avg_sentence_words

    @staticmethod
    def _hits(pats, text):
        return [m.group(0) for p in pats for m in p.finditer(text)]

    def strip_scores(self, text: str) -> tuple[str, list[str]]:
        """Remove whole sentences that contain a score. Returns (text, removed sentences)."""
        keep, removed = [], []
        for sent in _SENT.split(text.strip()):
            (removed if self._hits(self.score, sent) else keep).append(sent)
        return " ".join(s for s in keep if s).strip(), removed

    def check(self, reply: str, decision: Optional[SafetyDecision] = None) -> GuardResult:
        action = decision.action if decision else Action.CONTINUE
        if decision and decision.fixed_reply:
            return GuardResult(ok=True, text=decision.fixed_reply, action="show_fixed")

        text = normalize_quotes(reply).strip()   # "I’m a therapist" must match like "I'm a therapist"
        violations, warnings = [], []

        # 1. Scores: strip the sentences; if nothing useful is left, regenerate.
        text, removed = self.strip_scores(text)
        if removed:
            violations.append(f"score_leak (stripped {len(removed)} sentence(s))")

        # 2. Hard content checks.
        for name, pats in (("harm_content", self.harm), ("medical", self.medical), ("impersonation", self.imp)):
            hits = self._hits(pats, text)
            if hits:
                violations.append(f"{name}: {hits[:3]}")

        # 3. Crisis support replies must carry the helpline sentence (auto-fix).
        if action == Action.CRISIS_SUPPORT:
            support = "You can call SOS at 1767 or mindline at 1771 any time."
            if "1767" not in text or "1771" not in text:
                text = (text + " " + support).strip()
                warnings.append("support line appended")

        # 4. Plain-language lint.
        words = len(re.findall(r"[A-Za-z0-9']+", text))
        sents = [s for s in _SENT.split(text) if s.strip()]
        avg = words / max(1, len(sents))
        limit = self.max_words.get(action, 70)
        if words > limit:
            warnings.append(f"too_long: {words} words > {limit}")
        if avg > self.max_avg:
            warnings.append(f"long_sentences: avg {avg:.1f} words")
        idioms = [i for i in IDIOMS if i in text.lower()]
        if idioms:
            warnings.append(f"idioms: {idioms}")
        if self._hits(self.sarcasm, text):
            warnings.append("sarcasm marker")

        hard = [v for v in violations if not v.startswith("score_leak")]
        if hard or not text:
            return GuardResult(ok=False, text=SAFE_FALLBACK, action="regenerate", violations=violations,
                               warnings=warnings, regenerate_hint=self._hint(violations, warnings))
        if warnings and any(w.split(":")[0] in ("too_long", "long_sentences", "idioms", "sarcasm marker") for w in warnings):
            return GuardResult(ok=True, text=text, action="regenerate", violations=violations,
                               warnings=warnings, regenerate_hint=self._hint(violations, warnings))
        return GuardResult(ok=True, text=text, action="show", violations=violations, warnings=warnings)

    @staticmethod
    def _hint(violations, warnings) -> str:
        tips = []
        joined = " ".join(violations + warnings)
        if "score_leak" in joined:
            tips.append("Do not give any score, grade, or right/wrong mark.")
        if "harm_content" in joined:
            tips.append("Do not mention any method of self-harm.")
        if "medical" in joined:
            tips.append("Do not diagnose or give medical advice.")
        if "impersonation" in joined:
            tips.append("Say you are an AI. Never say you are a human or a professional.")
        if "too_long" in joined or "long_sentences" in joined:
            tips.append("Use fewer words and shorter sentences.")
        if "idioms" in joined or "sarcasm" in joined:
            tips.append("Use literal words only. No idioms or sarcasm.")
        return " ".join(tips)
