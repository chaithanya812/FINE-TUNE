"""Auto-improve loop - the "train again if it didn't get better" controller.

This is the heart of the original pitch: don't just train once. Train, measure,
and if the model didn't hit the target metric, escalate the recipe (more epochs,
more data, bigger adapter) and try again - keeping the best result. Stops early
the moment the target is reached, or after max_attempts.

Note: fine-tuning quality is driven mostly by DATA and TRAINING TIME, so the
escalation adds both, rather than just fiddling with the learning rate.
"""
from __future__ import annotations
import gc
from dataclasses import replace

import torch

from . import hyperparam
from .data import load_data
from .trainer import train
from .evaluate import evaluate_base_vs_tuned


def _escalation(model_id: str) -> list[dict]:
    """Escalating recipes, informed by Tinker's SFT sweep (see LEARNINGS_FROM_TINKER.md):
    raise the LoRA rank (16 -> 32 -> 64) and use their recommended LR, staying below the
    1e-3 divergence zone. Attempt 1 is the base config as given."""
    lr = hyperparam.recommended_lr(model_id)
    return [
        {},                                                        # 1: as configured
        {"epochs": 3.0, "n_train": 1200, "lora_r": 32, "lr": lr},  # 2: more data + Tinker rank/LR
        {"epochs": 4.0, "n_train": 2000, "lora_r": 64, "lr": lr},  # 3: more data + higher rank
    ]


def _metric(result: dict) -> float:
    if result["task"] == "classification":
        return result["after"]["accuracy"]
    return result["after"].get("avg_judge_score") or 0.0


def auto_improve(base_cfg, target=0.85, max_attempts=3, step_cb=None, event_cb=None) -> dict:
    """Train/evaluate repeatedly until `target` metric is hit or attempts run out.

    step_cb  receives per-step {step,max_steps,loss,epoch} during training.
    event_cb receives attempt-level events {type: "attempt"|"attempt_done", ...}.
    """
    history, best = [], None
    escalation = _escalation(base_cfg.model_id)

    for i in range(max_attempts):
        overrides = escalation[i] if i < len(escalation) else escalation[-1]
        cfg = replace(base_cfg, run_name=f"{base_cfg.run_name}_a{i + 1}", **overrides)
        cfg.save()
        if event_cb:
            event_cb({"type": "attempt", "attempt": i + 1, "max": max_attempts, "overrides": overrides})

        bundle = load_data(cfg)
        adapter = train(cfg, bundle, progress_cb=step_cb)
        result = evaluate_base_vs_tuned(cfg, bundle, adapter)
        m = _metric(result)

        history.append({"attempt": i + 1, "metric": m, "run_name": cfg.run_name, "overrides": overrides})
        if best is None or m > best["metric"]:
            best = {"attempt": i + 1, "metric": m, "run_name": cfg.run_name, "result": result, "adapter": adapter}
        if event_cb:
            event_cb({"type": "attempt_done", "attempt": i + 1, "metric": m,
                      "target": target, "hit": m >= target})

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if m >= target:
            break

    return {"best": best, "history": history, "target": target,
            "hit_target": best is not None and best["metric"] >= target}
