"""Unit tests for the confidence-aware robustness runner (finetune_studio/robustness.py).

Uses a MOCK predict_fn so the metamorphic-fail / banking / severity / calibration logic
is proven without a GPU. Runnable with pytest OR directly:
    .\\.venv\\Scripts\\python.exe tests\\test_robustness.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from finetune_studio import perturb, robustness  # noqa: E402
from server import db  # noqa: E402


def _fresh_store():
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="ft_rob_"))
    db.STORE, db.DB_PATH, db.BLOBS = tmp, tmp / "meta.db", tmp / "blobs"
    db.DATASET_BLOBS, db.EVALSET_BLOBS = db.BLOBS / "datasets", db.BLOBS / "evalsets"
    db.RUN_BLOBS, db.EXPORTS = db.BLOBS / "runs", tmp / "exports"
    db._BLOB_ROOTS = {"datasets": db.DATASET_BLOBS, "evalsets": db.EVALSET_BLOBS}
    db._initialized = False
    db.init()


EVAL = [{"input": "please cancel my order", "target": "CANCEL"},
        {"input": "where is my package now", "target": "TRACK"}]


def _mock_predictor():
    cases = perturb.perturb_rows(EVAL, seed=0)
    expected_of = {c["input"]: c["expected"] for c in cases}
    up = perturb.perturb_text("please cancel my order", "uppercase", 0)   # confidently-wrong flip
    ws = perturb.perturb_text("please cancel my order", "whitespace", 0)  # wrong, low confidence

    def predict(text: str) -> dict:
        if text == up:
            return {"label": "TRACK", "confidence": 0.95}
        if text == ws:
            return {"label": "TRACK", "confidence": 0.40}
        return {"label": expected_of.get(text, "?"), "confidence": 0.85}

    return predict, up, ws


def test_metamorphic_flip_severity_and_banking():
    _fresh_store()
    predict, up, ws = _mock_predictor()
    r1 = robustness.run_robustness("pr", EVAL, predict, labels=["CANCEL", "TRACK"], seed=0)

    assert r1["n_failures"] >= 2, "the uppercase + whitespace flips must fail"
    assert r1["severity_1"] >= 1, "the confidently-wrong flip is severity 1"
    assert r1["newly_banked"] >= 2, "failures are banked"

    # severity ranking: worst first, and a high-confidence wrong beats a low-confidence one
    assert r1["results"][0]["severity"] == 1
    assert r1["results"][0]["confidence"] >= 0.9
    sevs = [x["severity"] for x in r1["results"]]
    assert sevs == sorted(sevs), "results must be sorted worst-first"

    # calibration + coverage present
    assert r1["calibration"]["ece"] is not None
    cov = {(c["label"], c["perturbation"]): c for c in r1["coverage_matrix"]}
    assert cov[("CANCEL", "uppercase")]["fail"] >= 1


def test_banked_failure_refails_on_next_run():
    _fresh_store()
    predict, up, ws = _mock_predictor()
    robustness.run_robustness("pr", EVAL, predict, labels=["CANCEL", "TRACK"], seed=0)
    # next run: the manufactured failures are already banked (not re-added) and re-fail
    r2 = robustness.run_robustness("pr", EVAL, predict, labels=["CANCEL", "TRACK"], seed=0)
    assert r2["newly_banked"] == 0, "the bank must not double-add"
    assert r2["bank_regressed"] >= 2, "banked failures must re-fail on the next run"


def test_abstention_reported_separately():
    _fresh_store()

    def predict(text: str) -> dict:
        return {"label": "HELP", "confidence": 0.8}

    r = robustness.run_robustness("pa", [{"input": "reset my password", "target": "HELP"}],
                                  predict, labels=["HELP"], seed=0,
                                  abstention_probes=["what is my dog's name"])
    assert len(r["abstention"]) == 1 and r["abstention"][0]["input"] == "what is my dog's name"
    assert all("dog" not in x["input"] for x in r["results"]), "OOD probes must not count into accuracy"


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
