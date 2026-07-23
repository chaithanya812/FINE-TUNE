"""Synthetic data generation — the "teacher" (distillation) path.

Uses the user's Gemini key to generate diverse {input, target} training rows from a
plain-English task goal + a few optional seeds. Includes a token-cost estimate (for
the approval gate), batching, dedup, and light quality filtering. This is how a user
with no dataset still gets a trained model.
"""
from __future__ import annotations
import json
import os

from finetune_studio import dataio

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _model() -> str:
    """Active teacher model (set in the admin page). Read live on every call."""
    from server import settings
    return settings.current_gemini_model()


def available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def estimate(n: int, task: str = "classification", labels=None) -> dict:
    """Rough token-cost estimate so the user can approve before we spend anything."""
    out_tokens = n * (25 if task == "classification" else 60)
    calls = max(1, (n + 24) // 25)
    in_tokens = calls * (350 + (len(", ".join(labels)) if labels else 0))
    total = out_tokens + in_tokens
    model = _model()
    return {
        "n": n,
        "est_tokens": total,
        "est_calls": calls,
        "model": model,
        "note": f"{model} free tier usually covers this at no cost.",
    }


def _prompt(task: str, goal: str, n: int, labels=None, seeds=None) -> str:
    lines = [f"You are generating {n} diverse, realistic TRAINING examples for a small model.",
             f"TASK GOAL: {goal}", ""]
    if task == "classification" and labels:
        lines += [
            f"Each example is a short, natural user message and its correct label.",
            f"Use ONLY these labels and spread examples roughly evenly across them: {', '.join(labels)}.",
            'Return a JSON array of objects: {"input": "<user message>", "target": "<one label>"}.',
        ]
    elif task == "tone":
        lines += ['Each example is an input text and the same meaning rewritten in the target style/tone.',
                  'Return a JSON array of {"input": "<original text>", "target": "<rewritten text>"}.']
    elif task == "extraction":
        lines += ['Each example is messy input and the structured data to pull out of it.',
                  'Return a JSON array of {"input": "<messy text>", "target": "<compact JSON string>"}.']
    else:
        lines += ['Each example is a user message and an ideal assistant reply.',
                  'Return a JSON array of {"input": "<user message>", "target": "<ideal reply>"}.']
    if seeds:
        lines.append("\nMatch the style of these seed examples:")
        for s in seeds[:5]:
            lines.append(json.dumps({"input": s.get("input", ""), "target": s.get("target", "")}))
    lines.append("\nVary phrasing and cover edge cases. Output ONLY the JSON array, no prose or code fences.")
    return "\n".join(lines)


def _clean(arr) -> list[dict]:
    out = []
    for r in arr:
        if isinstance(r, dict) and str(r.get("input", "")).strip():
            out.append({"input": str(r["input"]).strip(), "target": str(r.get("target", "")).strip()})
    return out


def _salvage(s: str) -> list[dict]:
    """Pull complete {...} objects out of a (possibly truncated) array."""
    objs, depth, start = [], 0, None
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objs.append(json.loads(s[start:i + 1]))
                except Exception:
                    pass
                start = None
    return objs


def _parse(txt: str) -> list[dict]:
    """Robust JSON-array parse: tolerates code fences AND truncated responses
    (Gemini 2.5 Flash thinking can cut the array short — we salvage what's there)."""
    import re
    txt = (txt or "").strip()
    if "```" in txt:
        m = re.search(r"```(?:json)?\s*(.*?)```", txt, re.S)
        txt = m.group(1).strip() if m else txt.replace("```json", "").replace("```", "").strip()
    a = txt.find("[")
    if a < 0:
        return _clean(_salvage(txt))          # no array bracket — try loose objects
    b = txt.rfind("]")
    if b > a:
        try:
            return _clean(json.loads(txt[a:b + 1]))
        except Exception:
            pass
    return _clean(_salvage(txt[a:]))          # truncated array — salvage complete objects


def _client():
    from openai import OpenAI
    return OpenAI(base_url=GEMINI_BASE, api_key=os.environ["GEMINI_API_KEY"])


def generate(task: str, goal: str, n: int = 50, labels=None, seeds=None,
             temperature: float = 0.9, progress_cb=None) -> dict:
    """Generate up to n {input, target} rows. Returns {rows, n, errors}."""
    if not available():
        return {"rows": [], "n": 0, "errors": ["No GEMINI_API_KEY set — add it to .env."]}
    client = _client()
    model = _model()
    rows: list[dict] = []
    errors: list[str] = []
    per_call = 25
    guard = 0
    while len(rows) < n and guard < (n // per_call) + 4:
        guard += 1
        want = min(per_call, n - len(rows))
        try:
            resp = client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=4000,
                messages=[{"role": "user", "content": _prompt(task, goal, want, labels, seeds)}])
            batch = _parse(resp.choices[0].message.content)
            if not batch:
                errors.append("a batch returned no parseable examples")
            rows.extend(batch)
            rows = dataio.dedup(rows)
        except Exception as e:
            errors.append(str(e))
            break
        if progress_cb:
            progress_cb({"made": len(rows), "target": n})
    # light quality filter: drop absurdly short/long inputs
    rows = [r for r in rows if 1 <= len(r["input"]) <= 2000][:n]
    return {"rows": rows, "n": len(rows), "errors": errors}
