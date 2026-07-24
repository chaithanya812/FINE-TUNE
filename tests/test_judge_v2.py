"""Unit tests for judge v2 (finetune_studio/judge_v2.py) — pure pairwise logic with a
mock verdict_fn (no API), plus the DB judgment cache. Runnable with pytest OR directly:
    .\\.venv\\Scripts\\python.exe tests\\test_judge_v2.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from finetune_studio import judge_v2  # noqa: E402
from server import db  # noqa: E402


def _fresh_store():
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="ft_jv2_"))
    db.STORE, db.DB_PATH, db.BLOBS = tmp, tmp / "meta.db", tmp / "blobs"
    db.DATASET_BLOBS, db.EVALSET_BLOBS = db.BLOBS / "datasets", db.BLOBS / "evalsets"
    db.RUN_BLOBS, db.EXPORTS = db.BLOBS / "runs", tmp / "exports"
    db._BLOB_ROOTS = {"datasets": db.DATASET_BLOBS, "evalsets": db.EVALSET_BLOBS}
    db._initialized = False
    db.init()


def test_position_bias_becomes_tie():
    # a judge that ALWAYS says "A" (pure position bias) must be caught -> tie
    r = judge_v2.pairwise_swapped("q", "g", "reply-a", "reply-b", lambda i, g, a, b: "A")
    assert r["winner"] == "tie" and r["consistent"] is False


def test_content_preference_is_consistent():
    def vf(i, g, a, b):
        if a == "GOOD":
            return "A"
        if b == "GOOD":
            return "B"
        return "tie"
    r = judge_v2.pairwise_swapped("q", "g", "GOOD", "bad", vf)
    assert r["winner"] == "a" and r["consistent"] is True


def test_win_rates_ci_and_consistency():
    pairs = [{"winner": "a", "consistent": True}] * 7 + [{"winner": "b", "consistent": True}] * 3
    wr = judge_v2.win_rates(pairs)
    assert wr["a_wins"] == 7 and wr["b_wins"] == 3 and wr["ties"] == 0
    assert abs(wr["a_win_rate"] - 0.7) < 1e-9
    assert wr["lo"] <= 0.7 <= wr["hi"]
    assert wr["consistency"] == 1.0


def test_run_pairwise_flags_verbosity():
    samples = [{"input": f"q{i}", "tuned": "a genuinely long and detailed helpful reply",
                "base": "no"} for i in range(8)]

    def vf(i, g, a, b):                       # a verbosity-biased judge: longer wins
        return "A" if len(a) >= len(b) else "B"

    r = judge_v2.run_pairwise(None, samples, "tuned", "base", cap=8, verdict_fn=vf)
    assert r["a_wins"] == 8 and r["a_win_rate"] == 1.0
    assert r["mean_len_a"] > r["mean_len_b"]
    assert r["verbosity_warning"] is True


def test_anchors_detect_stable_and_broken_judges():
    def honest(i, g, a, b):                   # prefers whichever reply matches the reference
        return "A" if a == g else "B" if b == g else "tie"
    good = judge_v2.score_anchors(None, "generation", verdict_fn=honest)
    assert good["n"] >= 4 and good["great_wins"] == good["n"] and good["unstable"] is False

    broken = judge_v2.score_anchors(None, "generation", verdict_fn=lambda i, g, a, b: "B")
    assert broken["unstable"] is True         # terrible reply wins -> judge is untrustworthy


def test_judgment_cache_roundtrip():
    _fresh_store()
    assert db.cache_get("h1") is None
    db.cache_put("h1", "A", "modelX")
    assert db.cache_get("h1") == "A"
    db.cache_put("h1", "B", "modelX")          # overwrite same key
    assert db.cache_get("h1") == "B"


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
