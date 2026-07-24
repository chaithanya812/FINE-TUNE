"""Confidence-aware robustness autotest + permanent regression bank (Phase 3).

Instead of asking a teacher model to invent edge cases (flaky, unbounded), this runs
DETERMINISTIC label-preserving perturbations over the project's REAL eval rows and
checks that the prediction doesn't flip — a metamorphic test, no judge involved. It
also reads how CONFIDENT the model was, so we can surface the scariest failures first
(confidently wrong) and measure calibration (ECE). Every failure is banked and re-run
on every future version, so the suite only grows.

The runner takes a pluggable `predict_fn(text) -> {"label", "confidence"}` so the core
logic is testable without a GPU; model_service.classify_with_confidence is the real one.
"""
from __future__ import annotations

from collections import defaultdict

from finetune_studio import perturb, stats
from server import db

HIGH_CONFIDENCE = 0.70   # a wrong answer at/above this is "confidently wrong" = severity 1

# Out-of-domain probes: a router/classifier has no honest answer to these. Until a
# model can say "none", we report how it behaves on them SEPARATELY (never counted
# into accuracy) — the immortal "what is my dog's name" test.
ABSTENTION_PROBES = [
    "what is my dog's name",
    "write me a poem about the ocean",
    "what is 17 times 23",
    "who won the world cup in 1998",
    "tell me a joke about penguins",
]


def _label_eq(pred: str, expected: str) -> bool:
    p, e = (pred or "").strip().lower(), (expected or "").strip().lower()
    return bool(e) and (p == e or e in p)


def _severity(ok: bool, conf) -> int:
    if ok:
        return 3
    if conf is not None and conf >= HIGH_CONFIDENCE:
        return 1          # confidently wrong — the worst kind
    return 2              # wrong, but not confidently


def run_robustness(project_id: str, eval_rows: list[dict], predict_fn,
                   labels: list[str] | None = None, seed: int = 0,
                   types: list[str] | None = None,
                   abstention_probes: list[str] | None = None) -> dict:
    """Perturb real rows, catch prediction flips, bank failures, re-run the bank.

    `predict_fn(text)` must return {"label": str, "confidence": float|None}.
    """
    cases = perturb.perturb_rows(eval_rows, seed=seed, types=types)

    results, confidences, correct_flags = [], [], []
    coverage: dict[tuple[str, str], dict] = defaultdict(lambda: {"n": 0, "fail": 0})
    for c in cases:
        pred = predict_fn(c["input"])
        label, conf = pred.get("label"), pred.get("confidence")
        ok = _label_eq(label, c["expected"])
        sev = _severity(ok, conf)
        results.append({"input": c["input"], "expected": c["expected"],
                        "predicted": label, "confidence": conf, "ok": ok,
                        "perturbation": c["perturbation"], "severity": sev,
                        "source_input": c["source_input"]})
        if conf is not None:
            confidences.append(conf)
            correct_flags.append(ok)
        cell = coverage[(c["expected"], c["perturbation"])]
        cell["n"] += 1
        cell["fail"] += 0 if ok else 1

    # bank every metamorphic failure (dedup — the bank only grows)
    newly_banked = 0
    for r in results:
        if not r["ok"]:
            if db.bank_failure(project_id, r["input"], r["expected"], origin="perturbation",
                               perturbation=r["perturbation"], severity=r["severity"]):
                newly_banked += 1

    # re-run the EXISTING bank (regression) — new-vs-regressed
    bank_results = []
    for t in db.list_tests(project_id):
        pred = predict_fn(t["input"])
        ok = _label_eq(pred.get("label"), t["expected"])
        bank_results.append({"id": t["id"], "input": t["input"], "expected": t["expected"],
                             "predicted": pred.get("label"), "ok": ok,
                             "origin": t["origin"], "severity": t["severity"]})

    # abstention / out-of-domain probes — reported SEPARATELY, never folded into accuracy
    abstention = []
    for probe in (abstention_probes or []):
        pred = predict_fn(probe)
        abstention.append({"input": probe, "predicted": pred.get("label"),
                           "confidence": pred.get("confidence")})

    n = len(results)
    n_fail = sum(1 for r in results if not r["ok"])
    calibration = stats.ece(confidences, correct_flags) if confidences else {"ece": None, "n": 0, "bins": []}
    coverage_matrix = [
        {"label": lab, "perturbation": pt, "n": v["n"], "fail": v["fail"],
         "fail_rate": round(v["fail"] / v["n"], 3) if v["n"] else 0.0}
        for (lab, pt), v in sorted(coverage.items())
    ]
    results.sort(key=lambda r: (r["severity"], -(r["confidence"] or 0.0)))
    return {
        "project_id": project_id,
        "n_cases": n,
        "n_failures": n_fail,
        "pass_rate": round((n - n_fail) / n, 3) if n else 1.0,
        "severity_1": sum(1 for r in results if r["severity"] == 1),
        "calibration": calibration,
        "coverage_matrix": coverage_matrix,
        "results": results[:50],
        "bank_size": len(bank_results),
        "newly_banked": newly_banked,
        "bank_regressed": sum(1 for b in bank_results if not b["ok"]),
        "bank_results": bank_results[:50],
        "abstention": abstention,
    }
