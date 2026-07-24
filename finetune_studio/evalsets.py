"""Frozen, leakage-guarded evaluation sets.

A score is only comparable across versions if every version is graded on the SAME
exam, and the exam is only honest if its questions aren't near-copies of the study
material. This module carves a stratified hold-out out of a dataset and quarantines
any eval row that is a near-duplicate of a training row (memorization, not skill).

  * build_frozen_eval(rows, ...) -> {eval, train, quarantined, meta}
        stratified by class, near-dup quarantined, golden rows always kept in eval.
  * held_out_train(all_rows, eval_rows) -> training rows with the frozen eval held
        out even after the dataset is edited (used when re-training a later version).

Similarity is cheap and dependency-free: normalized-token Jaccard, confirmed by a
difflib character ratio only for plausible twins. Either metric >= threshold (~0.85)
counts as a twin. Embeddings can replace this later without changing callers.
"""
from __future__ import annotations

import difflib
import random
import re
from collections import defaultdict

_TOKEN = re.compile(r"[a-z0-9]+")
_GOLD_TRUE = {"1", "true", "yes", "y", "golden", "gold"}


def _norm_tokens(text) -> set[str]:
    return set(_TOKEN.findall(str(text).lower()))


def jaccard(a, b) -> float:
    ta, tb = _norm_tokens(a), _norm_tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def ratio(a, b) -> float:
    return difflib.SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def is_near_dup(a, b, threshold: float = 0.85) -> bool:
    """Twin if token-Jaccard >= threshold, or (for plausible pairs) difflib ratio >=."""
    j = jaccard(a, b)
    if j >= threshold:
        return True
    if j >= 0.5:                       # only pay for the char-level check on close pairs
        return ratio(a, b) >= threshold
    return False


def is_golden(row: dict) -> bool:
    v = row.get("golden")
    if v is True:
        return True
    return str(v).strip().lower() in _GOLD_TRUE if v is not None else False


def _has_twin(text, others, threshold: float) -> bool:
    return any(is_near_dup(text, o, threshold) for o in others)


def build_frozen_eval(rows: list[dict], task: str = "classification",
                      target_frac: float = 0.2, seed: int = 42,
                      threshold: float = 0.85, min_train_per_class: int = 1) -> dict:
    """Carve a stratified, leakage-guarded eval set out of `rows`.

    Golden rows are always placed in eval (human-approved exam questions) and are
    exempt from quarantine — but a golden row that twins a training row is flagged
    in meta.golden_leak_warn so the user can see it. Non-golden eval rows that twin
    ANY training row are evicted back to training (they'd only measure memorization).
    """
    rng = random.Random(seed)
    clean = [dict(r) for r in rows if str(r.get("input", "")).strip()]

    # 1. stratified candidate pick -------------------------------------------------
    eval_cand: list[dict] = []
    train: list[dict] = []
    if task == "classification":
        by_class: dict[str, list[dict]] = defaultdict(list)
        for r in clean:
            by_class[str(r.get("target", ""))].append(r)
        for _label, items in sorted(by_class.items()):
            items = items[:]
            rng.shuffle(items)
            golden = [r for r in items if is_golden(r)]
            rest = [r for r in items if not is_golden(r)]
            n = len(items)
            k = min(round(n * target_frac), max(0, n - min_train_per_class))
            eval_cand.extend(golden + rest[:k])
            train.extend(rest[k:])
    else:
        items = clean[:]
        rng.shuffle(items)
        golden = [r for r in items if is_golden(r)]
        rest = [r for r in items if not is_golden(r)]
        k = min(round(len(items) * target_frac), max(0, len(items) - 1))
        eval_cand = golden + rest[:k]
        train = rest[k:]

    # 2. quarantine non-golden eval rows that twin a training row (iterate to fixpoint)
    quarantined: list[dict] = []
    eval_rows = eval_cand
    for _ in range(4):
        train_inputs = [r.get("input", "") for r in train]
        keep, moved = [], []
        for r in eval_rows:
            if not is_golden(r) and _has_twin(r.get("input", ""), train_inputs, threshold):
                moved.append(r)
            else:
                keep.append(r)
        if not moved:
            eval_rows = keep
            break
        quarantined.extend(moved)
        train.extend(moved)
        eval_rows = keep

    # 3. dedup within eval (keep first; twins go to training) ----------------------
    deduped: list[dict] = []
    for r in eval_rows:
        seen = [x.get("input", "") for x in deduped]
        if is_golden(r) or not _has_twin(r.get("input", ""), seen, threshold):
            deduped.append(r)
        else:
            quarantined.append(r)
            train.append(r)
    eval_rows = deduped

    # 4. meta ----------------------------------------------------------------------
    golden_leak = [r for r in eval_rows
                   if is_golden(r) and _has_twin(r.get("input", ""),
                                                 [t.get("input", "") for t in train], threshold)]
    per_class = defaultdict(int)
    for r in eval_rows:
        per_class[str(r.get("target", ""))] += 1
    meta = {
        "task": task, "threshold": threshold, "seed": seed, "target_frac": target_frac,
        "n_eval": len(eval_rows), "n_train": len(train), "n_quarantined": len(quarantined),
        "n_golden": sum(1 for r in eval_rows if is_golden(r)),
        "per_class_eval": dict(per_class),
        "golden_leak_warn": len(golden_leak),
    }
    return {"eval": eval_rows, "train": train, "quarantined": quarantined, "meta": meta}


def get_or_create_frozen(project_id: str, rows: list[dict], task: str,
                         source_dataset_version_id: str | None = None,
                         target_frac: float = 0.2, seed: int = 42,
                         threshold: float = 0.85) -> dict:
    """Return the project's frozen eval set, creating it ONCE if it doesn't exist.

    Created once and reused forever so every version is graded on a byte-identical
    exam. Stored immutably via db.add_eval_set (content-addressed blob).
    """
    from server import db
    existing = db.list_eval_sets(project_id)
    if existing:
        return existing[0]
    built = build_frozen_eval(rows, task=task, target_frac=target_frac,
                              seed=seed, threshold=threshold)
    return db.add_eval_set(project_id, built["eval"], meta=built["meta"],
                           created_from_dataset_version=source_dataset_version_id)


def held_out_train(all_rows: list[dict], eval_rows: list[dict],
                   threshold: float = 0.85) -> list[dict]:
    """Return training rows with the frozen eval set held out.

    Used when re-training a LATER dataset version against an eval set frozen from an
    earlier one: any current row that is a near-duplicate of a frozen eval row (or an
    exact input match) is dropped so the exam stays unseen.
    """
    eval_inputs = [r.get("input", "") for r in eval_rows]
    exact = {str(r.get("input", "")).strip() for r in eval_rows}
    out = []
    for r in all_rows:
        inp = str(r.get("input", "")).strip()
        if not inp or inp in exact:
            continue
        if _has_twin(inp, eval_inputs, threshold):
            continue
        out.append(r)
    return out
