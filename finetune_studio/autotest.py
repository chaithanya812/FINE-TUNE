"""AI auto-test — the agent's own quality check (Stage 6, "Thing 2").

Generates hard edge cases for the task with the teacher model, runs them through the
tuned model, scores each (correct / partial / wrong), and returns a scorecard +
verdict the user can act on ("production-ready" vs "let's fix what's failing").
"""
from __future__ import annotations
import os

from finetune_studio import synth


def _edge_prompt(task: str, goal: str, labels, n: int) -> str:
    lines = [f"Generate {n} HARD, tricky edge-case TEST inputs for this task:",
             f"GOAL: {goal}", ""]
    if task == "classification" and labels:
        lines += [
            f"Make them genuinely tricky: ambiguous, multi-intent, angry, terse, or misspelled.",
            f"Valid labels: {', '.join(labels)}.",
            'Return a JSON array of {"input": "<tricky message>", "target": "<the single best label>"}.',
        ]
    else:
        lines += ['Make them ambiguous or adversarial.',
                  'Return a JSON array of {"input": "<tricky input>", "target": "<ideal answer>"}.']
    lines.append("Output ONLY the JSON array.")
    return "\n".join(lines)


def generate_edge_cases(task: str, goal: str, labels=None, n: int = 10) -> list[dict]:
    if not synth.available():
        return []
    from openai import OpenAI
    client = OpenAI(base_url=synth.GEMINI_BASE, api_key=os.environ["GEMINI_API_KEY"])
    model = synth._model()
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.9, max_tokens=4000,
            messages=[{"role": "user", "content": _edge_prompt(task, goal, labels, n)}])
        return synth._parse(resp.choices[0].message.content)
    except Exception:
        return []


def _load_json_obj(text: str):
    """Pull the first {...} object out of possibly-fenced/prose text."""
    import json
    s = (text or "").strip()
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        return json.loads(s[a:b + 1])
    except Exception:
        return None


def _json_match(pred: str, expected: str) -> str:
    """Real extraction metric: fraction of expected fields the prediction got right."""
    pe, pp = _load_json_obj(expected), _load_json_obj(pred)
    if not isinstance(pe, dict) or not pe:
        return "partial" if (pred or "").strip() else "wrong"
    if not isinstance(pp, dict):
        return "wrong"
    hits = sum(1 for k, v in pe.items()
               if k in pp and str(pp[k]).strip().lower() == str(v).strip().lower())
    frac = hits / len(pe)
    return "correct" if frac == 1 else "partial" if frac >= 0.5 else "wrong"


def _score(task: str, pred: str, expected: str) -> str:
    if task == "extraction":
        return _json_match(pred, expected)
    p, e = (pred or "").strip().lower(), (expected or "").strip().lower()
    if task == "classification":
        if e and (p == e or e in p):
            return "correct"
        return "wrong"
    # open-ended: we can't auto-grade perfectly; treat a non-empty on-topic reply as partial
    return "partial" if p else "wrong"


def _verdict(score: float, n: int) -> str:
    if n == 0:
        return "Couldn't generate test cases (is the teacher API set up?)."
    if score >= 0.9:
        return "Model is production-ready."
    if score >= 0.7:
        return "Solid, with a few gaps worth fixing."
    return "Not there yet — more training data on the weak cases would help."


def run_autotest(run_name: str, task: str, goal: str, labels=None, n: int = 10, cases=None) -> dict:
    """Score the tuned model on generated (or supplied) edge cases."""
    from server import model_service
    cases = cases or generate_edge_cases(task, goal, labels, n)
    results, correct = [], 0
    for c in cases:
        try:
            pred = model_service.chat_with_run(run_name, c["input"], use_adapter=True).get("reply", "")
        except Exception as e:
            pred = f"(error: {e})"
        verdict = _score(task, pred, c.get("target", ""))
        correct += 1 if verdict == "correct" else 0
        results.append({"input": c["input"], "expected": c.get("target"), "got": pred, "verdict": verdict})
    total = len(results)
    score = correct / total if total else 0.0
    return {"results": results, "correct": correct, "total": total,
            "score": round(score, 3), "verdict": _verdict(score, total)}
