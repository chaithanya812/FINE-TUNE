"""Unit tests for finetune_studio/evalsets.py.

Runnable with pytest OR directly:
    .\\.venv\\Scripts\\python.exe tests\\test_evalsets.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from finetune_studio import evalsets as es  # noqa: E402


def _distinct(prefix, n, target):
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
             "hotel", "india", "juliet", "kilo", "lima"]
    return [{"input": f"{prefix} {words[i]} {i}", "target": target} for i in range(n)]


def test_is_near_dup():
    assert es.is_near_dup("please cancel my subscription now",
                          "please cancel my subscription now!!!", 0.85)
    assert not es.is_near_dup("hello there how are you", "goodbye see you later", 0.85)


def test_seeded_twin_is_quarantined_out():
    # GREET/BYE are distinct; CANCEL is two twins (identical tokens) -> one must be evicted
    rows = (_distinct("greeting", 5, "GREET")
            + _distinct("farewell", 5, "BYE")
            + [{"input": "please cancel my subscription now", "target": "CANCEL"},
               {"input": "please cancel my subscription now!!!", "target": "CANCEL"}])
    res = es.build_frozen_eval(rows, task="classification", target_frac=0.4,
                               seed=1, threshold=0.85)
    assert res["meta"]["n_quarantined"] >= 1, "the twin must be quarantined"
    # the surviving eval set contains no 'cancel' twin
    assert not any("cancel my subscription" in r["input"] for r in res["eval"])
    # and both cancel rows ended up available for training instead
    assert sum("cancel my subscription" in r["input"] for r in res["train"]) == 2


def test_no_eval_row_twins_any_train_row():
    rows = (_distinct("greeting", 6, "GREET")
            + _distinct("farewell", 6, "BYE")
            + [{"input": "reset my password please", "target": "HELP"},
               {"input": "reset my password please now", "target": "HELP"},
               {"input": "totally different help request here", "target": "HELP"}])
    res = es.build_frozen_eval(rows, task="classification", seed=3, threshold=0.85)
    train_inputs = [r["input"] for r in res["train"]]
    for r in res["eval"]:
        if es.is_golden(r):
            continue
        assert not es._has_twin(r["input"], train_inputs, 0.85), \
            f"leaked eval row twins training: {r['input']!r}"


def test_stratified_across_classes():
    rows = _distinct("greeting", 6, "GREET") + _distinct("farewell", 6, "BYE")
    res = es.build_frozen_eval(rows, task="classification", target_frac=0.34, seed=5)
    pc = res["meta"]["per_class_eval"]
    assert pc.get("GREET", 0) >= 1 and pc.get("BYE", 0) >= 1
    # every class keeps at least one training example
    train_classes = {r["target"] for r in res["train"]}
    assert {"GREET", "BYE"} <= train_classes


def test_golden_rows_always_in_eval_and_flagged_if_leaky():
    rows = _distinct("topic", 5, "A") + [
        {"input": "shared phrase about billing issues", "target": "A"},
        {"input": "shared phrase about billing issues", "target": "A", "golden": "true"},
    ]
    res = es.build_frozen_eval(rows, task="classification", target_frac=0.2, seed=2)
    golden_in_eval = [r for r in res["eval"] if es.is_golden(r)]
    assert len(golden_in_eval) == 1, "golden row must be kept in eval"
    # its non-golden twin is in training -> golden leak is surfaced, not silently dropped
    assert res["meta"]["golden_leak_warn"] >= 1


def test_determinism():
    rows = _distinct("x", 12, "A") + _distinct("y", 12, "B")
    a = es.build_frozen_eval(rows, seed=9)
    b = es.build_frozen_eval(rows, seed=9)
    assert {r["input"] for r in a["eval"]} == {r["input"] for r in b["eval"]}


def test_held_out_train_drops_frozen_eval():
    eval_rows = [{"input": "cancel my plan today", "target": "CANCEL"}]
    all_rows = [
        {"input": "cancel my plan today", "target": "CANCEL"},        # exact -> dropped
        {"input": "cancel my plan today!", "target": "CANCEL"},       # twin -> dropped
        {"input": "upgrade my plan today", "target": "UPGRADE"},      # different -> kept
    ]
    train = es.held_out_train(all_rows, eval_rows, threshold=0.85)
    inputs = [r["input"] for r in train]
    assert inputs == ["upgrade my plan today"]


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
