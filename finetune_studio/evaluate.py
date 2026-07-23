"""Evaluation - the "did it get better?" module.

Loads the base model ONCE, attaches the trained adapter, and scores both on the
same held-out set (VRAM-friendly: `disable_adapter()` gives us the base behaviour
without loading a second copy). This is what produces your before/after numbers.

It returns more than a score: alongside accuracy it collects REAL example
predictions (the same customer message, what the base model guessed vs what the
tuned model guesses, and which is correct) plus a per-class breakdown - so the UI
and the agent can *show* the improvement, not just report a number.
"""
from __future__ import annotations
import torch
from peft import PeftModel

from .data import build_prompt_messages
from .trainer import load_tokenizer, build_model


def _generate(model, tokenizer, messages, max_new_tokens=24) -> str:
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _match_label(text: str, labels: list[str]) -> str:
    t = text.strip().lower()
    for lab in labels:                       # exact
        if t == lab.lower():
            return lab
    for lab in labels:                       # substring (lenient, applied to both models)
        if lab.lower() in t:
            return lab
    return "unknown"


# ----------------------------------------------------------------- classification
def _classify_preds(model, tokenizer, bundle) -> list[str]:
    """One predicted label per held-out example (in eval_df order)."""
    preds = []
    for _, r in bundle.eval_df.iterrows():
        msgs = build_prompt_messages(r["input"], bundle.task, bundle.labels)
        preds.append(_match_label(_generate(model, tokenizer, msgs, 16), bundle.labels))
    return preds


def _cls_metrics(golds, preds, labels) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy": float(accuracy_score(golds, preds)),
        "macro_f1": float(f1_score(golds, preds, average="macro", labels=labels, zero_division=0)),
        "n": len(golds),
    }


def _pick_samples(inputs, golds, base_preds, tuned_preds, k=8) -> list[dict]:
    """Choose human-legible examples to showcase: the model's *wins* first (tuned
    fixed what the base got wrong), then a couple of honest regressions, then a few
    where both were right - so the user sees real proof, good and bad."""
    wins, regressions, both_right, both_wrong = [], [], [], []
    for inp, g, b, t in zip(inputs, golds, base_preds, tuned_preds):
        rec = {"input": inp, "gold": g, "base": b, "tuned": t,
               "base_ok": b == g, "tuned_ok": t == g}
        if t == g and b != g:
            wins.append(rec)
        elif b == g and t != g:
            regressions.append(rec)
        elif b == g and t == g:
            both_right.append(rec)
        else:
            both_wrong.append(rec)
    ordered = wins + regressions[:2] + both_right + both_wrong
    return ordered[:k]


def _per_class(golds, base_preds, tuned_preds, k=8) -> list[dict]:
    """Per-label support + base/tuned accuracy for the classes present in the eval
    set (largest first) - the 'which topics improved' view."""
    from collections import defaultdict
    total, base_ok, tuned_ok = defaultdict(int), defaultdict(int), defaultdict(int)
    for g, b, t in zip(golds, base_preds, tuned_preds):
        total[g] += 1
        base_ok[g] += int(b == g)
        tuned_ok[g] += int(t == g)
    rows = [{"label": lab, "support": n,
             "base_acc": base_ok[lab] / n, "tuned_acc": tuned_ok[lab] / n}
            for lab, n in total.items()]
    rows.sort(key=lambda r: r["support"], reverse=True)
    return rows[:k]


# --------------------------------------------------------------------- generation
def _gen_replies(model, tokenizer, inputs, task="generation") -> list[str]:
    return [_generate(model, tokenizer, build_prompt_messages(i, task, []), 120) for i in inputs]


def _strong_prompt_messages(input_text, task) -> list[dict]:
    """The 3rd arm of the eval: the base model with a genuinely GOOD prompt.
    If prompting alone matches the fine-tune, the honest answer is 'don't fine-tune'."""
    from . import tasks
    msgs = tasks.build_prompt_messages(task, input_text, [])
    msgs[0]["content"] += (
        "\nBe specific, warm, and well-structured. Answer in 1-3 tight sentences that "
        "directly solve the user's need — no filler, no generic lists."
    )
    return msgs


def _judge_avg(cfg, inputs, replies):
    from .judge import score_reply
    scores = [score_reply(cfg, i, rep) for i, rep in zip(inputs, replies)]
    scores = [s for s in scores if s is not None]
    return (sum(scores) / len(scores)) if scores else None


# ------------------------------------------------------------------------- driver
def evaluate_base_vs_tuned(cfg, bundle, adapter_dir, progress_cb=None) -> dict:
    """Score base (adapter off) vs tuned (adapter on) on the same eval set, and
    collect real example predictions for the UI / agent to showcase."""
    tokenizer = load_tokenizer(cfg)
    base = build_model(cfg, for_training=False)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()

    inputs = [str(r["input"]) for _, r in bundle.eval_df.iterrows()]

    if bundle.task == "classification":
        golds = [str(r[bundle.label_field]) for _, r in bundle.eval_df.iterrows()]

        if progress_cb:
            progress_cb({"phase": "eval_after"})
        tuned_preds = _classify_preds(model, tokenizer, bundle)

        if progress_cb:
            progress_cb({"phase": "eval_before"})
        with model.disable_adapter():
            base_preds = _classify_preds(model, tokenizer, bundle)

        after = _cls_metrics(golds, tuned_preds, bundle.labels)
        before = _cls_metrics(golds, base_preds, bundle.labels)
        result = {
            "task": "classification",
            "before": before,
            "after": after,
            "delta_accuracy": after["accuracy"] - before["accuracy"],
            "samples": _pick_samples(inputs, golds, base_preds, tuned_preds),
            "per_class": _per_class(golds, base_preds, tuned_preds),
            "labels": bundle.labels,
        }
    else:
        from .judge import available
        judge_on = available(cfg)
        note = None if judge_on else "no JUDGE_API_KEY set"

        if progress_cb:
            progress_cb({"phase": "eval_after"})
        tuned_replies = _gen_replies(model, tokenizer, inputs, bundle.task)

        if progress_cb:
            progress_cb({"phase": "eval_before"})
        with model.disable_adapter():
            base_replies = _gen_replies(model, tokenizer, inputs, bundle.task)

        # 3rd arm (guarded — never breaks the run): base model + a STRONG prompt.
        # A fine-tune only counts if it beats good prompting too.
        prompted = None
        prompted_replies = None
        try:
            if progress_cb:
                progress_cb({"phase": "eval_prompted"})
            with model.disable_adapter():
                prompted_replies = [
                    _generate(model, tokenizer, _strong_prompt_messages(i, bundle.task), 120)
                    for i in inputs
                ]
            prompted = {"avg_judge_score": _judge_avg(cfg, inputs, prompted_replies) if judge_on else None,
                        "n": len(inputs)}
        except Exception as e:
            print(f"[eval] prompted-baseline arm skipped: {e}")
            prompted_replies = None

        after = {"avg_judge_score": _judge_avg(cfg, inputs, tuned_replies) if judge_on else None,
                 "n": len(inputs), "note": note}
        before = {"avg_judge_score": _judge_avg(cfg, inputs, base_replies) if judge_on else None,
                  "n": len(inputs), "note": note}
        b, a = before["avg_judge_score"], after["avg_judge_score"]
        samples = [{"input": i, "base": bl, "tuned": tl}
                   for i, bl, tl in zip(inputs, base_replies, tuned_replies)]
        if prompted_replies:
            for s, pr in zip(samples, prompted_replies):
                s["prompted"] = pr
        result = {
            "task": "generation",
            "before": before,
            "after": after,
            "prompted": prompted,
            "delta_judge": (a - b) if (a is not None and b is not None) else None,
            "samples": samples[:6],
        }

    # Free VRAM so the "chat with your model" panel can load afterwards.
    del model, base
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
