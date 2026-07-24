"""Statistics for honest evals — pure functions, stdlib only, unit-tested.

Every score the studio reports should come with an uncertainty and a sample size,
so "89%" becomes "89% (95% CI 78-96, n=45)" and a v1-vs-v2 win isn't called real
until it clears a significance test. Two tools cover the whole studio:

  * bootstrap_ci  — percentile bootstrap for ANY statistic (accuracy, judge mean).
  * mcnemar       — exact paired test for classifier version-vs-version, where both
                    models were scored on the SAME items (the only fair comparison).

No numpy/scipy: keeps this importable anywhere and easy to hand-check.
"""
from __future__ import annotations

import random
from math import comb
from typing import Callable, Sequence


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank percentile, p in [0, 1], over indices [0, len-1]."""
    if not sorted_vals:
        return float("nan")
    idx = int(round(p * (len(sorted_vals) - 1)))
    idx = min(len(sorted_vals) - 1, max(0, idx))
    return sorted_vals[idx]


def bootstrap_ci(values: Sequence[float], statistic: Callable[[Sequence[float]], float] = mean,
                 n_resamples: int = 2000, ci: float = 0.95, seed: int = 0) -> dict:
    """Percentile bootstrap CI for `statistic` over `values`.

    Resamples the items (with replacement) n_resamples times and takes the
    empirical (1-ci)/2 and (1+ci)/2 percentiles of the resampled statistic.
    Deterministic given `seed`. Returns {point, lo, hi, n}.
    """
    vals = list(values)
    n = len(vals)
    if n == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    point = statistic(vals)
    if n == 1:
        return {"point": point, "lo": point, "hi": point, "n": 1}
    rng = random.Random(seed)
    resampled = []
    for _ in range(n_resamples):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        resampled.append(statistic(sample))
    resampled.sort()
    alpha = (1.0 - ci) / 2.0
    return {"point": point,
            "lo": _percentile(resampled, alpha),
            "hi": _percentile(resampled, 1.0 - alpha),
            "n": n}


def accuracy_ci(correct: Sequence, n_resamples: int = 2000, ci: float = 0.95, seed: int = 0) -> dict:
    """Bootstrap CI for accuracy. `correct` is a sequence of truthy/falsy per-item flags."""
    flags = [1.0 if c else 0.0 for c in correct]
    d = bootstrap_ci(flags, mean, n_resamples, ci, seed)
    return {"accuracy": d["point"], "lo": d["lo"], "hi": d["hi"], "n": d["n"], "ci": ci}


def mean_ci(values: Sequence[float], n_resamples: int = 2000, ci: float = 0.95, seed: int = 0) -> dict:
    """Bootstrap CI for a mean (e.g. judge scores)."""
    d = bootstrap_ci(values, mean, n_resamples, ci, seed)
    return {"mean": d["point"], "lo": d["lo"], "hi": d["hi"], "n": d["n"], "ci": ci}


def mcnemar(a_correct: Sequence, b_correct: Sequence) -> dict:
    """Exact McNemar test for two classifiers scored on the SAME items (paired).

    Only the discordant pairs matter:
      b = items where A is right and B is wrong
      c = items where A is wrong and B is right
    Under H0 (no difference) each discordant pair is a fair coin, so the two-sided
    p-value is the exact binomial tail. Exact (not chi-square) because our eval
    sets are small. Returns {b, c, n_discordant, p_value, statistic}.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("mcnemar needs paired inputs of equal length")
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if (not x) and y)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p_value": 1.0, "statistic": 0.0}
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    p_value = min(1.0, 2.0 * tail)
    chi2 = (abs(b - c) - 1) ** 2 / n           # continuity-corrected, for reference
    return {"b": b, "c": c, "n_discordant": n, "p_value": p_value, "statistic": chi2}


def compare_classifiers(a_correct: Sequence, b_correct: Sequence,
                        n_resamples: int = 2000, ci: float = 0.95, seed: int = 0) -> dict:
    """Full paired comparison of two classifier versions on the same eval items."""
    a = accuracy_ci(a_correct, n_resamples, ci, seed)
    b = accuracy_ci(b_correct, n_resamples, ci, seed)
    return {"a": a, "b": b, "delta": b["accuracy"] - a["accuracy"],
            "mcnemar": mcnemar(a_correct, b_correct), "n": a["n"]}


def fmt_ci(point: float, lo: float, hi: float, n: int, pct: bool = True) -> str:
    """Human string: '89% (95% CI 78-96, n=45)'  /  '7.2 (95% CI 6.1-8.0, n=30)'."""
    if n == 0:
        return "n/a (n=0)"
    if pct:
        return f"{point*100:.0f}% (95% CI {lo*100:.0f}-{hi*100:.0f}, n={n})"
    return f"{point:.2f} (95% CI {lo:.2f}-{hi:.2f}, n={n})"
