"""Data ingestion + validation.

Turns an uploaded CSV / JSONL / pasted blob into normalized {input, target} rows,
auto-detecting which columns are which, and computes the stats the UI needs
(class balance, thin classes, train/val/test split, dedup).
"""
from __future__ import annotations
import csv
import io
import json
import random
from collections import Counter

INPUT_NAMES = ["input", "instruction", "text", "message", "utterance", "prompt", "question"]
TARGET_NAMES = ["target", "label", "intent", "category", "output", "response", "answer", "completion"]


def _pick(cols, names, override=None):
    if override and override in cols:
        return override
    low = {c.lower(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    return None


def _read_csv(text, errors):
    try:
        reader = csv.DictReader(io.StringIO(text))
        return list(reader), list(reader.fieldnames or [])
    except Exception as e:
        errors.append(f"CSV parse error: {e}")
        return [], []


def _read_jsonl(text, errors):
    records = []
    if text[:1] == "[":
        try:
            records = [r for r in json.loads(text) if isinstance(r, dict)]
        except Exception as e:
            errors.append(f"JSON parse error: {e}")
    else:
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if isinstance(r, dict):
                    records.append(r)
            except Exception as e:
                errors.append(f"line {i}: {e}")
    cols: list[str] = []
    for r in records:
        for k in r:
            if k not in cols:
                cols.append(k)
    return records, cols


def parse_upload(text: str, filename: str = "", input_col=None, target_col=None) -> dict:
    """Parse an upload into {input, target} rows. Returns rows + detected columns + errors."""
    text = (text or "").strip()
    if not text:
        return {"rows": [], "columns": [], "errors": ["The file is empty."]}
    errors: list[str] = []
    if filename.lower().endswith(".jsonl") or filename.lower().endswith(".json") or text[:1] in "{[":
        records, columns = _read_jsonl(text, errors)
    else:
        records, columns = _read_csv(text, errors)

    ic = _pick(columns, INPUT_NAMES, input_col)
    tc = _pick(columns, TARGET_NAMES, target_col)
    rows = []
    if ic is None:
        errors.append(f"Couldn't find an input column (looked for {INPUT_NAMES}). "
                      f"Columns found: {columns}. Tell me which column is the input.")
    else:
        for rec in records:
            inp = str(rec.get(ic, "")).strip()
            tgt = str(rec.get(tc, "")).strip() if tc else ""
            if inp:
                rows.append({"input": inp, "target": tgt})
    return {"rows": rows, "columns": columns, "input_col": ic, "target_col": tc, "errors": errors}


def dedup(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        key = (r.get("input", "").strip().lower(), r.get("target", "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def stats(rows: list[dict], task: str = "classification") -> dict:
    n = len(rows)
    out: dict = {"n": n}
    if task == "classification":
        counts = Counter(r["target"] for r in rows if r.get("target"))
        out["n_classes"] = len(counts)
        out["classes"] = [{"label": c, "count": k} for c, k in counts.most_common()]
        if counts:
            mx = max(counts.values())
            thin = [c for c, k in counts.items() if k < 20 and k <= max(1, mx * 0.3)]
            out["thin_classes"] = thin
            out["balance_warning"] = (
                f"Thin classes (few examples): {', '.join(thin)} — consider generating more."
                if thin else None)
        missing = sum(1 for r in rows if not r.get("target"))
        if missing:
            out["missing_targets"] = missing
    return out


def split(rows: list[dict], seed: int = 42, ratios=(0.8, 0.1, 0.1)) -> dict:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n = len(rows)
    ntr, nva = int(n * ratios[0]), int(n * ratios[1])
    return {"train": rows[:ntr], "val": rows[ntr:ntr + nva], "test": rows[ntr + nva:]}


def split_counts(rows, seed=42, ratios=(0.8, 0.1, 0.1)) -> dict:
    s = split(rows, seed, ratios)
    return {k: len(v) for k, v in s.items()}
