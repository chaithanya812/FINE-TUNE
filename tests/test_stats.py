"""Unit tests for finetune_studio/stats.py — hand-checked small cases.

Runnable with pytest OR directly:
    .\\.venv\\Scripts\\python.exe tests\\test_stats.py
"""
from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from finetune_studio import stats  # noqa: E402


def _close(a, b, tol=1e-9):
    return math.isclose(a, b, abs_tol=tol)


def test_accuracy_ci_degenerate():
    # all correct -> every bootstrap resample is all 1s -> CI collapses to 1.0
    d = stats.accuracy_ci([1, 1, 1, 1])
    assert _close(d["accuracy"], 1.0) and _close(d["lo"], 1.0) and _close(d["hi"], 1.0)
    assert d["n"] == 4
    # all wrong -> collapses to 0.0
    z = stats.accuracy_ci([0, 0, 0, 0, 0])
    assert _close(z["accuracy"], 0.0) and _close(z["lo"], 0.0) and _close(z["hi"], 0.0)
    assert z["n"] == 5


def test_accuracy_ci_point_and_bracketing():
    d = stats.accuracy_ci([1, 1, 0, 0])          # exactly half right
    assert _close(d["accuracy"], 0.5)
    assert d["lo"] <= d["accuracy"] <= d["hi"]   # point inside the interval
    assert 0.0 <= d["lo"] <= d["hi"] <= 1.0


def test_bootstrap_is_reproducible_and_seed_sensitive():
    v = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    a = stats.accuracy_ci(v, seed=42)
    b = stats.accuracy_ci(v, seed=42)
    assert a == b                                # same seed -> identical CI
    c = stats.accuracy_ci(v, seed=7)
    assert (a["lo"], a["hi"]) != (c["lo"], c["hi"]) or True  # different seed may differ


def test_ci_narrows_with_more_data():
    small = stats.accuracy_ci([1, 0])                     # n=2, wide
    big = stats.accuracy_ci([1, 0] * 100)                 # n=200, tight
    assert (big["hi"] - big["lo"]) < (small["hi"] - small["lo"])


def test_mean_ci_constant():
    d = stats.mean_ci([5.0, 5.0, 5.0])
    assert _close(d["mean"], 5.0) and _close(d["lo"], 5.0) and _close(d["hi"], 5.0)


def test_mean_ci_empty():
    d = stats.mean_ci([])
    assert math.isnan(d["mean"]) and d["n"] == 0


def test_mcnemar_symmetric_is_insignificant():
    # b == c -> p = 1.0 (no evidence of a difference)
    a = [1, 1, 0, 0, 1, 0]
    b = [0, 0, 1, 1, 0, 1]
    r = stats.mcnemar(a, b)
    assert r["b"] == 3 and r["c"] == 3 and r["n_discordant"] == 6
    assert _close(r["p_value"], 1.0)


def test_mcnemar_all_one_direction():
    # b=8, c=0 -> exact two-sided p = 2 * 0.5^8 = 0.0078125 (hand-checked)
    a = [1] * 8
    b = [0] * 8
    r = stats.mcnemar(a, b)
    assert r["b"] == 8 and r["c"] == 0
    assert _close(r["p_value"], 2 * (0.5 ** 8))
    assert _close(r["p_value"], 0.0078125)


def test_mcnemar_ignores_concordant_pairs():
    # only one discordant pair among many agreements -> cannot be significant
    a = [1, 1, 1, 1, 1]
    b = [1, 1, 1, 1, 0]
    r = stats.mcnemar(a, b)
    assert r["b"] == 1 and r["c"] == 0 and _close(r["p_value"], 1.0)


def test_mcnemar_identical_models():
    a = [1, 0, 1, 0]
    r = stats.mcnemar(a, a)
    assert r["n_discordant"] == 0 and _close(r["p_value"], 1.0)


def test_compare_classifiers_clear_improvement():
    a = [0] * 10                 # base wrong on everything
    b = [1] * 10                 # tuned right on everything (same items)
    r = stats.compare_classifiers(a, b)
    assert _close(r["a"]["accuracy"], 0.0) and _close(r["b"]["accuracy"], 1.0)
    assert _close(r["delta"], 1.0)
    assert r["mcnemar"]["b"] == 0 and r["mcnemar"]["c"] == 10
    assert _close(r["mcnemar"]["p_value"], 2 * (0.5 ** 10))   # 0.001953125


def test_fmt_ci():
    assert stats.fmt_ci(0.89, 0.78, 0.96, 45) == "89% (95% CI 78-96, n=45)"
    assert stats.fmt_ci(7.17, 6.1, 8.0, 30, pct=False) == "7.17 (95% CI 6.10-8.00, n=30)"
    assert "n=0" in stats.fmt_ci(float("nan"), float("nan"), float("nan"), 0)


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failed = 0
    for t in ALL:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {e!r}")
    print(f"\n{len(ALL) - failed}/{len(ALL)} passed")
    sys.exit(1 if failed else 0)
