"""Text normalisation shared by the rules, the classifier and the training scripts.

Two variants:
  * normalize_for_rules(): lower-case, apostrophes removed, leetspeak undone,
    euphemisms expanded. Regex patterns in lexicons.py are written for this form
    ("dont", "im", "cant" ...).
  * normalize_for_model(): keeps case and punctuation (the transformer is cased)
    but removes tricks that hide words (zero-width characters, stretched letters,
    euphemisms). Training and inference MUST use the same function.
"""
from __future__ import annotations

import re
import unicodedata

_ZERO_WIDTH = re.compile(r"[​-‏⁠﻿­]")
_REPEAT = re.compile(r"(.)\1{2,}")           # "soooo" -> "soo"
_SPACES = re.compile(r"\s+")
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "`": "'"})
_LEET_MAP = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_TOKEN = re.compile(r"[a-z0-9@$]+")

# Euphemisms and spellings used to get around filters. Applied case-insensitively.
EUPHEMISMS: list[tuple[str, str]] = [
    (r"\bun-?aliv(e|ing|ed)\b", "kill"),        # "unalive myself" -> "kill myself"
    (r"\bkms\b", "kill myself"),
    (r"\bkys\b", "kill yourself"),
    (r"\bsewer[\s-]?slide\b", "suicide"),
    (r"\bsu[i1!*]c[i1!*]d[e3]\b", "suicide"),
    (r"\bs[u*]ic[i*]dal\b", "suicidal"),
    (r"\bself[\s-]+harm", "selfharm"),
    (r"\bk[i1!*]ll\b", "kill"),
]
_EUPH = [(re.compile(p, re.IGNORECASE), r) for p, r in EUPHEMISMS]


def _base(text: str, squeeze: int = 2) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = _ZERO_WIDTH.sub("", t)
    t = t.translate(_QUOTES)
    t = _REPEAT.sub(r"\1" * squeeze, t)
    return t


def _undo_leet(t: str) -> str:
    def fix(m: re.Match) -> str:
        tok = m.group(0)
        has_alpha = any(c.isalpha() for c in tok)
        has_trick = any(c in "0134576@$" for c in tok)
        # Only rewrite mixed tokens like "d1e" or "k!ll"; leave "1767" and "9am" alone.
        if has_alpha and has_trick and len(tok) >= 3 and not tok[0].isdigit():
            return tok.translate(_LEET_MAP)
        return tok

    return _TOKEN.sub(fix, t)


def normalize_for_rules(text: str, squeeze: int = 2) -> str:
    """squeeze=2 turns "tooo" into "too"; squeeze=1 turns "dieee" into "die". The rules try both."""
    t = _base(text, squeeze).lower()
    t = _undo_leet(t)
    for pat, rep in _EUPH:
        t = pat.sub(rep, t)
    t = t.replace("'", "")                      # "don't" -> "dont", "i'm" -> "im"
    t = re.sub(r"[^\w\s.?!,]", " ", t)          # drop emoji/symbols for matching
    return _SPACES.sub(" ", t).strip()


def normalize_for_model(text: str) -> str:
    t = _base(text)
    for pat, rep in _EUPH:
        t = pat.sub(rep, t)
    return _SPACES.sub(" ", t).strip()


def normalize_quotes(text: str) -> str:
    """Curly quotes and backticks -> plain ASCII, nothing else changed (used on companion replies)."""
    return (text or "").translate(_QUOTES)


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text or ""))
