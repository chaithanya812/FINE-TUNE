"""finetune_studio - a tiny end-to-end LoRA/QLoRA fine-tuning + evaluation engine.

Pipeline:  data -> train (LoRA/QLoRA) -> evaluate (base vs tuned) -> chat.
Shared by both the local runner (scripts/run_local.py) and the web UI.
"""
__all__ = ["data", "trainer", "evaluate", "inference", "judge"]
