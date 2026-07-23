"""LoRA learning-rate scaling — adopted from Thinking Machines' Tinker Cookbook.

Their empirical result (tinker_cookbook/hyperparam_utils.py + research notes):

    LR(model) = lr_base * M_LoRA * (2000 / hidden_size) ** P

with lr_base=5e-5, M_LoRA=10 for LoRA (1 for full fine-tuning), and a model-family
exponent P. They report <0.5% regret vs exhaustive LR sweeps. See
../LEARNINGS_FROM_TINKER.md for context and attribution.
"""
from __future__ import annotations

LR_BASE = 5e-5
M_LORA = 10.0
FAMILY_EXPONENT = {"qwen": 0.0775, "llama": 0.781}

# Hidden sizes for the models we support locally / via Colab export.
HIDDEN_SIZE = {
    "Qwen/Qwen2.5-0.5B-Instruct": 896,
    "Qwen/Qwen2.5-1.5B-Instruct": 1536,
    "Qwen/Qwen2.5-3B-Instruct": 2048,
    "Qwen/Qwen2.5-7B-Instruct": 3584,
    "meta-llama/Llama-3.2-1B-Instruct": 2048,
    "meta-llama/Llama-3.2-3B-Instruct": 3072,
}

# Tinker's SFT-sweep rank guidance: default 32; 64-128 helps more at higher LR;
# 8-16 fine for simple adaptations. LR is independent of rank.
DEFAULT_RANK = 32
# lr >= 1e-3 diverged in ~72/80 of their diverging runs — keep below this.
DIVERGENCE_LR = 1e-3


def _family(model_id: str) -> str:
    m = model_id.lower()
    if "qwen" in m:
        return "qwen"
    if "llama" in m:
        return "llama"
    return "qwen"  # sensible default for unknown Qwen-like models


def recommended_lr(model_id: str, hidden_size: int | None = None, is_lora: bool = True) -> float:
    """Tinker's LoRA LR formula. Returns a safe LR below the divergence zone."""
    h = hidden_size or HIDDEN_SIZE.get(model_id, 2048)
    p = FAMILY_EXPONENT[_family(model_id)]
    m = M_LORA if is_lora else 1.0
    lr = LR_BASE * m * (2000.0 / h) ** p
    return min(lr, DIVERGENCE_LR * 0.9)
