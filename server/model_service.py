"""Shared model service - loads a trained (base + adapter) model and chats with it.

Centralized so the /api/chat endpoint AND the AI agent use ONE cached model (critical on
a 4GB GPU - two caches would try to hold two copies and OOM).
"""
from __future__ import annotations
import json

from config import Config
from finetune_studio.data import build_prompt_messages
from finetune_studio.inference import load_for_chat, chat as chat_fn

_cache: dict = {"key": None, "model": None, "tok": None, "meta": None}


def _load_run_meta(run_name: str):
    cfg = Config.preset("local_4gb", run_name=run_name)
    cfg_path = cfg.output_dir / "config.json"
    if cfg_path.exists():
        saved = json.loads(cfg_path.read_text())
        for k in ("task", "label_field", "model_id", "mode"):
            if k in saved:
                setattr(cfg, k, saved[k])
    labels_path = cfg.output_dir / "labels.json"
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else []
    return cfg, labels


def adapter_exists(run_name: str) -> bool:
    return Config.preset("local_4gb", run_name=run_name).adapter_dir.exists()


def unload():
    """Drop the cached chat model and free its VRAM. Call before starting a
    training job so a warm playground cache doesn't OOM the 4 GB card."""
    import gc
    _cache.update(key=None, model=None, tok=None, meta=None)
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _get_model(run_name: str):
    cfg, labels = _load_run_meta(run_name)
    adapter = cfg.adapter_dir
    key = str(adapter)
    if _cache["key"] != key:
        import gc
        import torch
        _cache["model"] = None
        gc.collect()
        torch.cuda.empty_cache()
        model, tok = load_for_chat(cfg, adapter_dir=adapter if adapter.exists() else None)
        _cache.update(key=key, model=model, tok=tok, meta=(cfg, labels))
    return _cache["model"], _cache["tok"], *_cache["meta"]


def chat_with_run(run_name: str, message: str, use_adapter: bool = True) -> dict:
    model, tok, cfg, labels = _get_model(run_name)
    messages = build_prompt_messages(message, cfg.task, labels)
    max_new = 24 if cfg.task == "classification" else 200
    temp = 0.0 if cfg.task == "classification" else 0.7
    if use_adapter or not hasattr(model, "disable_adapter"):
        reply = chat_fn(model, tok, messages, max_new_tokens=max_new, temperature=temp)
    else:
        with model.disable_adapter():
            reply = chat_fn(model, tok, messages, max_new_tokens=max_new, temperature=temp)
    return {"reply": reply, "task": cfg.task}
