"""Unit tests for Phase 3 pure pieces: perturb.py, stats.ece, and the db regression bank.

Runnable with pytest OR directly:
    .\\.venv\\Scripts\\python.exe tests\\test_perturb.py
"""
from __future__ import annotations

import math
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from finetune_studio import perturb, stats  # noqa: E402
from server import db  # noqa: E402


def _fresh_store():
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="ft_p3_"))
    db.STORE, db.DB_PATH, db.BLOBS = tmp, tmp / "meta.db", tmp / "blobs"
    db.DATASET_BLOBS, db.EVALSET_BLOBS = db.BLOBS / "datasets", db.BLOBS / "evalsets"
    db.RUN_BLOBS, db.EXPORTS = db.BLOBS / "runs", tmp / "exports"
    db._BLOB_ROOTS = {"datasets": db.DATASET_BLOBS, "evalsets": db.EVALSET_BLOBS}
    db._initialized = False
    db.init()


# ------------------------------------------------------------------- perturb
def test_perturb_is_deterministic():
    t = "Please cancel my subscription right now"
    for ttype in perturb.PERTURBATION_TYPES:
        assert perturb.perturb_text(t, ttype, seed=1) == perturb.perturb_text(t, ttype, seed=1)


def test_perturbations_change_text_and_are_labelled():
    ps = perturb.perturbations("The mobile app keeps crashing on startup", seed=0)
    assert len(ps) >= 5
    for p in ps:
        assert p["input"] != "The mobile app keeps crashing on startup"
        assert p["type"] in perturb.PERTURBATION_TYPES


def test_uppercase_is_recoverable_but_changed():
    assert perturb.perturb_text("hello world", "uppercase") == "HELLO WORLD"
    assert perturb.perturb_text("Do not refund me", "contraction").lower() == "don't refund me"


def test_perturb_rows_carries_expected_label():
    rows = [{"input": "where is my order", "target": "ORDER_STATUS"}]
    cases = perturb.perturb_rows(rows, seed=3)
    assert cases and all(c["expected"] == "ORDER_STATUS" for c in cases)
    assert all(c["source_input"] == "where is my order" for c in cases)
    assert all("perturbation" in c for c in cases)


# ----------------------------------------------------------------------- ece
def test_ece_perfect_calibration():
    d = stats.ece([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0])
    assert math.isclose(d["ece"], 0.0, abs_tol=1e-9)


def test_ece_confidently_wrong_is_high():
    d = stats.ece([0.95, 0.95, 0.95, 0.95], [0, 0, 0, 0])
    assert math.isclose(d["ece"], 0.95, abs_tol=1e-9)


def test_ece_half_right_full_confidence():
    d = stats.ece([1.0, 1.0, 1.0, 1.0], [1, 0, 1, 0])
    assert math.isclose(d["ece"], 0.5, abs_tol=1e-9)


def test_ece_empty():
    d = stats.ece([], [])
    assert math.isnan(d["ece"]) and d["n"] == 0


# ------------------------------------------------------------- regression bank
def test_bank_dedup_and_severity_order():
    _fresh_store()
    a = db.bank_failure("p1", "shouting input", "ANGRY", origin="perturbation",
                        perturbation="uppercase", severity=1)
    assert a is not None
    dup = db.bank_failure("p1", "shouting input", "ANGRY")
    assert dup is None, "the bank must not double-add the same failing case"
    db.add_test("p1", "a milder case", "NEUTRAL", origin="manual", severity=3)
    bank = db.list_tests("p1")
    assert len(bank) == 2
    assert bank[0]["severity"] == 1 and bank[1]["severity"] == 3  # worst first
    assert bank[0]["origin"] == "perturbation" and bank[0]["perturbation"] == "uppercase"


def test_bank_only_grows_across_runs():
    _fresh_store()
    db.bank_failure("p2", "case one", "A")
    db.bank_failure("p2", "case two", "B")
    db.bank_failure("p2", "case one", "A")  # re-seen next run -> not re-added
    assert len(db.list_tests("p2")) == 2


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
