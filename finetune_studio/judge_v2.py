"""Judge v2 — de-biased, uncertainty-aware LLM judging (Phase 4).

The old judge gives a bare 1-10 that's easy to game and hard to defend: the teacher
grading its own students, position bias, verbosity bias, no error bars. v2:

  * Reference-guided PAIRWISE: given (input, gold, reply A, reply B) pick the better
    one. Every pair is judged TWICE with A/B swapped; if the verdict flips, the judge
    is position-biased on that pair, so it counts as a tie. Aggregate to win-rates
    with bootstrap CIs and a position-consistency score.
  * Anchors: fixed terrible-vs-great triples judged every run — if "terrible" ever
    beats "great", the judge itself is unstable (warn, don't trust the numbers).
  * Caching: verdicts are memoized in the DB by (model, input, replies), so re-running
    an eval is near-free.
  * Verbosity guard: report mean reply length per arm and flag a winner that's just
    systematically longer.
  * Jury: when a second judge family (Groq) is configured alongside Gemini, both are
    reported as available (mitigates the teacher-grades-its-own-students self-bias).

The pairwise core takes a pluggable verdict_fn, so it's unit-testable with no API.
"""
from __future__ import annotations

import hashlib
import json
import os

from . import stats

_REF = "Use this reference answer as the gold standard for a great reply:\n{gold}\n\n"
PAIRWISE_PROMPT = """You are comparing two assistant replies to the same user message.
{ref_block}Judge which reply is better for the user: more correct, complete, and appropriately
toned. Do NOT prefer a reply just because it is longer.

User message:
{input}

Reply A:
{a}

Reply B:
{b}

Respond with ONLY a JSON object: {{"winner": "A" | "B" | "tie", "reason": "<one line>"}}"""

# Fixed sanity anchors: a clearly terrible vs a clearly great reply. Judged every run.
ANCHORS = {
    "generation": [
        {"input": "How do I reset my password?",
         "terrible": "idk figure it out",
         "great": "Go to Settings -> Security -> Reset password, then check your email for the link."},
        {"input": "Has my order shipped yet?",
         "terrible": "no.",
         "great": "Yes - it shipped today and should arrive in 2-3 business days; here's your tracking link."},
        {"input": "Can I get a refund?",
         "terrible": "we don't do that ever, stop asking",
         "great": "Absolutely - refunds are available within 30 days. I can start that now; can you confirm your order number?"},
        {"input": "Do you support dark mode?",
         "terrible": "what's dark mode",
         "great": "Yes! Turn it on under Settings -> Appearance -> Dark; it can also follow your system theme."},
    ],
}


# ------------------------------------------------------------------ judge calls
def judge_families(cfg) -> list[tuple]:
    """Available (base_url, key, model, family) judge endpoints, primary first."""
    if cfg is None:
        return []
    from . import judge as j
    fams = []
    jkey = os.environ.get(cfg.judge_api_key_env)
    if jkey:
        fams.append((cfg.judge_base_url, jkey, cfg.judge_model, "groq"))
    gkey = os.environ.get("GEMINI_API_KEY")
    if gkey:
        from server import settings
        fams.append((j.GEMINI_BASE, gkey, settings.current_gemini_model(), "gemini"))
    return fams


def _call(base, key, model, prompt) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=base, api_key=key)
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content or ""


def _parse_winner(txt: str) -> str:
    try:
        obj = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        w = str(obj.get("winner", "tie")).strip().upper()
        return w if w in ("A", "B") else "tie"
    except Exception:
        return "tie"


def _hash(model, input, a, b) -> str:
    return hashlib.sha256(f"{model}\x00{input}\x00{a}\x00{b}".encode("utf-8")).hexdigest()


def make_verdict_fn(cfg, use_cache: bool = True):
    """Real reference-guided pairwise verdict via the primary judge family (cached)."""
    fams = judge_families(cfg)
    if not fams:
        return None
    base, key, model, _family = fams[0]

    def verdict(input: str, gold: str, a: str, b: str) -> str:
        h = _hash(model, input, a, b)
        if use_cache:
            from server import db
            hit = db.cache_get(h)
            if hit:
                return hit
        prompt = PAIRWISE_PROMPT.format(ref_block=_REF.format(gold=gold) if gold else "",
                                        input=input, a=a, b=b)
        try:
            w = _parse_winner(_call(base, key, model, prompt))
        except Exception as e:  # noqa: BLE001
            print(f"[judge-v2] error: {e}")
            w = "tie"
        if use_cache:
            from server import db
            db.cache_put(h, w, model)
        return w

    return verdict


# --------------------------------------------------------------- pairwise core
def pairwise_swapped(input: str, gold: str, a: str, b: str, verdict_fn) -> dict:
    """Judge (a vs b) in BOTH orders; a flipped verdict = position bias = tie."""
    v1 = verdict_fn(input, gold, a, b)          # A=a, B=b
    v2 = verdict_fn(input, gold, b, a)          # A=b, B=a
    w1 = "a" if v1 == "A" else "b" if v1 == "B" else "tie"
    w2 = "a" if v2 == "B" else "b" if v2 == "A" else "tie"   # map back (positions swapped)
    consistent = (w1 == w2)
    return {"winner": w1 if consistent else "tie", "consistent": consistent, "v1": w1, "v2": w2}


def win_rates(pairs: list[dict], seed: int = 0) -> dict:
    """Aggregate pairwise outcomes into A's win-rate + bootstrap CI + consistency."""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "a_wins": 0, "b_wins": 0, "ties": 0,
                "a_win_rate": None, "lo": None, "hi": None, "consistency": None}
    outcomes = [1.0 if p["winner"] == "a" else 0.0 if p["winner"] == "b" else 0.5 for p in pairs]
    ci = stats.bootstrap_ci(outcomes, seed=seed)
    return {"n": n,
            "a_wins": sum(1 for p in pairs if p["winner"] == "a"),
            "b_wins": sum(1 for p in pairs if p["winner"] == "b"),
            "ties": sum(1 for p in pairs if p["winner"] == "tie"),
            "a_win_rate": ci["point"], "lo": ci["lo"], "hi": ci["hi"],
            "consistency": sum(1 for p in pairs if p["consistent"]) / n}


def run_pairwise(cfg, samples: list[dict], arm_a: str = "tuned", arm_b: str = "base",
                 cap: int = 30, verdict_fn=None) -> dict:
    """Reference-guided pairwise eval of arm_a vs arm_b over stored samples.

    Each sample: {input, gold?, <arm_a>, <arm_b>}. Returns arm_a's win-rate with CI,
    position-swap consistency, and a verbosity-bias flag.
    """
    vf = verdict_fn or make_verdict_fn(cfg)
    if vf is None:
        return {"error": "no judge available"}
    used = [s for s in samples if s.get(arm_a) and s.get(arm_b)][:cap]
    pairs, la, lb = [], [], []
    for s in used:
        gold = s.get("gold") or s.get("target") or ""
        pairs.append(pairwise_swapped(str(s["input"]), str(gold), str(s[arm_a]), str(s[arm_b]), vf))
        la.append(len(str(s[arm_a])))
        lb.append(len(str(s[arm_b])))
    wr = win_rates(pairs)
    mean_la = sum(la) / len(la) if la else 0.0
    mean_lb = sum(lb) / len(lb) if lb else 0.0
    winner_longer = (wr["a_win_rate"] is not None and wr["a_win_rate"] > 0.5
                     and mean_la > 1.25 * mean_lb)
    return {"arm_a": arm_a, "arm_b": arm_b, **wr,
            "mean_len_a": round(mean_la, 1), "mean_len_b": round(mean_lb, 1),
            "verbosity_warning": bool(winner_longer),
            "families": [f[3] for f in judge_families(cfg)]}


def score_anchors(cfg, task: str = "generation", verdict_fn=None) -> dict:
    """Judge fixed terrible-vs-great anchors; if terrible ever wins, the judge is unstable."""
    vf = verdict_fn or make_verdict_fn(cfg)
    anchors = ANCHORS.get(task, [])
    if vf is None or not anchors:
        return {"n": 0, "great_wins": 0, "unstable": False, "results": []}
    results, great = [], 0
    for an in anchors:
        pw = pairwise_swapped(an["input"], an["great"], an["great"], an["terrible"], vf)
        won = pw["winner"] == "a"    # a = the great reply
        great += 1 if won else 0
        results.append({"input": an["input"], "great_won": won, "consistent": pw["consistent"]})
    return {"n": len(anchors), "great_wins": great,
            "unstable": great < len(anchors), "results": results}
