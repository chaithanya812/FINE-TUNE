"""Deterministic, label-preserving perturbations for metamorphic testing.

A robust classifier shouldn't change its answer because the user SHOUTED, added a
typo, or dropped a period. So we take REAL eval rows, apply small label-preserving
edits, and re-run the model: if the prediction flips, that's a deterministic FAIL —
no judge, no teacher, no flakiness. (Only label-preserving transforms live here;
meaning-changing edits like negation are out of scope so a flip is always a bug.)

Every transform is seeded by a stable sha256 of (seed, type, text), so the exact
same perturbations reproduce across processes and machines.
"""
from __future__ import annotations

import hashlib
import random
import re

_CONTRACTIONS = {
    r"\bdo not\b": "don't", r"\bcan not\b": "can't", r"\bcannot\b": "can't",
    r"\bi am\b": "i'm", r"\byou are\b": "you're", r"\bit is\b": "it's",
    r"\bthat is\b": "that's", r"\bi have\b": "i've", r"\bwill not\b": "won't",
    r"\bdid not\b": "didn't", r"\bdoes not\b": "doesn't", r"\bis not\b": "isn't",
}
_EMOJI = ["🙂", "🔥", "😡", "👍", "🤔", "😊"]


def _rng(seed: int, ttype: str, text: str) -> random.Random:
    h = hashlib.sha256(f"{seed}:{ttype}:{text}".encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


def _typo(text: str, rng: random.Random) -> str:
    if len(text) < 4:
        return text
    i = rng.randrange(1, len(text) - 2)          # swap two adjacent characters
    return text[:i] + text[i + 1] + text[i] + text[i + 2:]


def _uppercase(text: str, rng: random.Random) -> str:
    return text.upper()


def _lowercase(text: str, rng: random.Random) -> str:
    return text.lower()


def _randomcaps(text: str, rng: random.Random) -> str:
    return "".join(c.upper() if rng.random() < 0.35 else c for c in text)


def _punctuation(text: str, rng: random.Random) -> str:
    return text.rstrip(" .!?") + rng.choice(["!!!", "...", "?!", " ."])


def _emoji(text: str, rng: random.Random) -> str:
    return text + " " + rng.choice(_EMOJI)


def _filler(text: str, rng: random.Random) -> str:
    pre = rng.choice(["um, ", "like, ", "honestly, ", "so, ", "hey, "])
    post = rng.choice([" please", " thanks", " asap", ""])
    return pre + text + post


def _whitespace(text: str, rng: random.Random) -> str:
    return re.sub(r" ", "  ", text, count=1) + " "


def _contraction(text: str, rng: random.Random) -> str:
    out = text
    for pat, rep in _CONTRACTIONS.items():
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return out


TRANSFORMS = {
    "typo": _typo, "uppercase": _uppercase, "lowercase": _lowercase,
    "randomcaps": _randomcaps, "punctuation": _punctuation, "emoji": _emoji,
    "filler": _filler, "whitespace": _whitespace, "contraction": _contraction,
}
PERTURBATION_TYPES = list(TRANSFORMS)


def perturb_text(text: str, ttype: str, seed: int = 0) -> str:
    """Apply one named transform deterministically."""
    return TRANSFORMS[ttype](str(text), _rng(seed, ttype, str(text)))


def perturbations(text: str, seed: int = 0, types: list[str] | None = None) -> list[dict]:
    """All label-preserving perturbations of `text` that actually changed it.

    Returns [{type, input}]. `input` is the perturbed text; the expected label is
    the SAME as the original row's (that's the whole point of label-preserving).
    """
    text = str(text)
    out = []
    for t in (types or PERTURBATION_TYPES):
        p = perturb_text(text, t, seed)
        if p and p != text:
            out.append({"type": t, "input": p})
    return out


def perturb_rows(rows: list[dict], seed: int = 0, types: list[str] | None = None) -> list[dict]:
    """Expand real eval rows into a metamorphic test suite.

    Each output case carries its origin (source input + transform) and the expected
    label (unchanged), so a runner can flag any prediction flip as a FAIL.
    """
    cases = []
    for r in rows:
        src = str(r.get("input", ""))
        exp = r.get("target", "")
        for p in perturbations(src, seed=seed, types=types):
            cases.append({"input": p["input"], "expected": exp,
                          "perturbation": p["type"], "source_input": src})
    return cases
