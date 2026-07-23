"""Inference - load a (base + adapter) model and chat with it.

Used by the UI "chat with your fine-tuned model" panel and for quick manual checks.
Pass adapter_dir=None to talk to the plain base model for comparison.
"""
from __future__ import annotations
import torch
from peft import PeftModel

from .trainer import load_tokenizer, build_model


def load_for_chat(cfg, adapter_dir=None):
    tokenizer = load_tokenizer(cfg)
    model = build_model(cfg, for_training=False)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tokenizer


def chat(model, tokenizer, messages, max_new_tokens=200, temperature=0.7) -> str:
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0, temperature=temperature, top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
