"""System / GPU probe — powers the "your hardware" badge (Stage 1) and the live
VRAM bar (Stage 4). Uses torch's CUDA memory info; degrades gracefully with no GPU.
"""
from __future__ import annotations

from finetune_studio import catalog


def gpu_info() -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"cuda": False, "name": None, "total_gb": 0.0, "free_gb": 0.0, "used_gb": 0.0}
        free, total = torch.cuda.mem_get_info()
        return {
            "cuda": True,
            "name": torch.cuda.get_device_name(0),
            "total_gb": round(total / 1e9, 2),
            "free_gb": round(free / 1e9, 2),
            "used_gb": round((total - free) / 1e9, 2),
        }
    except Exception as e:
        return {"cuda": False, "name": None, "total_gb": 0.0, "free_gb": 0.0, "used_gb": 0.0, "error": str(e)}


def _badge(g: dict, info) -> str:
    if not g.get("cuda"):
        return "No CUDA GPU detected → use the Colab export path for training"
    label = info.label if info else "Qwen2.5 0.5B"
    return f"{g['total_gb']:.0f} GB GPU detected ({g['name']}) → {label} runs locally"


def system_summary() -> dict:
    g = gpu_info()
    # Base the recommendation on *free* VRAM so the suggestion is honest about what
    # can actually train right now (a 4 GB card realistically means the 0.5B).
    basis = (g.get("free_gb") or 0.0) if g.get("cuda") else 0.0
    rec = catalog.recommend(basis)
    info = catalog.get(rec)
    return {
        "gpu": g,
        "recommended_model": rec,
        "recommended_label": info.label if info else rec,
        "badge": _badge(g, info),
        "models": catalog.as_dicts(free_gb=basis if g.get("cuda") else 0.0),
    }
