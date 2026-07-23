"""Model catalog + a VRAM "will it fit?" estimator.

A small, curated list of base models we support, with the facts a non-technical
user needs to choose one (size, license, context length, what it's good at) and a
rough VRAM estimate so the UI can say "runs locally on your 4 GB" vs "use Colab".

The estimates are heuristics for QLoRA 4-bit training / 4-bit inference — good
enough to guide a choice, not exact. Always leave headroom on a 4 GB card.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass
class ModelInfo:
    id: str
    label: str
    params_b: float          # billions of parameters
    hidden_size: int         # used by hyperparam.recommended_lr
    context: int             # max context length (tokens)
    license: str
    strengths: str
    family: str              # "qwen" | "llama" (for the LR formula)


# Ordered small -> large. Local (4 GB) realistically means the 0.5B; the rest are
# offered for the Colab-export path or a bigger GPU.
CATALOG: list[ModelInfo] = [
    ModelInfo("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5 0.5B", 0.5, 896, 32768,
              "Apache-2.0", "Fast and tiny — great for classification & simple tasks", "qwen"),
    ModelInfo("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5 1.5B", 1.5, 1536, 32768,
              "Apache-2.0", "Balanced — better wording for generation / tone tasks", "qwen"),
    ModelInfo("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5 3B", 3.0, 2048, 32768,
              "Qwen Research License", "Strong quality — needs Colab on a 4 GB card", "qwen"),
    ModelInfo("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5 7B", 7.0, 3584, 32768,
              "Apache-2.0", "High quality — Colab / bigger GPU only", "qwen"),
    ModelInfo("meta-llama/Llama-3.2-1B-Instruct", "Llama 3.2 1B", 1.0, 2048, 131072,
              "Llama 3.2 Community License", "Long context, solid general chat", "llama"),
    ModelInfo("meta-llama/Llama-3.2-3B-Instruct", "Llama 3.2 3B", 3.0, 3072, 131072,
              "Llama 3.2 Community License", "Long context, stronger reasoning", "llama"),
]

_BY_ID = {m.id: m for m in CATALOG}


def get(model_id: str) -> ModelInfo | None:
    return _BY_ID.get(model_id)


def estimate_train_gb(m: ModelInfo) -> float:
    """Rough QLoRA (4-bit) training footprint in GB (weights + LoRA + paged optim + acts).

    Deliberately conservative: on this project a 4 GB card only reliably fits the
    0.5B (real runs ~2.3 GB); 1.5B+ OOMs and must go to Colab. The coefficient is
    tuned so the estimator reflects that hard-won reality, not paper minimums.
    """
    return round(m.params_b * 1.9 + 1.6, 1)


def estimate_infer_gb(m: ModelInfo) -> float:
    """Rough 4-bit inference footprint in GB."""
    return round(m.params_b * 0.7 + 0.8, 1)


def fits_local(m: ModelInfo, free_gb: float, mode: str = "train") -> bool:
    need = estimate_train_gb(m) if mode == "train" else estimate_infer_gb(m)
    return need <= free_gb


def recommend(free_gb: float) -> str:
    """Biggest model whose QLoRA training fits the free VRAM; else the smallest."""
    local = [m for m in CATALOG if fits_local(m, free_gb, "train")]
    if local:
        return max(local, key=lambda m: m.params_b).id
    return min(CATALOG, key=lambda m: m.params_b).id


def as_dicts(free_gb: float | None = None) -> list[dict]:
    out = []
    for m in CATALOG:
        d = asdict(m)
        d["est_train_gb"] = estimate_train_gb(m)
        d["est_infer_gb"] = estimate_infer_gb(m)
        if free_gb is not None:
            d["fits_local_train"] = fits_local(m, free_gb, "train")
            d["fits_local_infer"] = fits_local(m, free_gb, "infer")
        out.append(d)
    return out
