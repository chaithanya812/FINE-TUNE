"""LLM-as-judge - scores an open-ended support reply 1-10 using a strong model.

Used only for the "generation" task (open-ended replies have no exact answer to
match). Talks to any OpenAI-compatible endpoint - Groq, OpenRouter, Google AI
Studio, etc. Set the key in the env var named by cfg.judge_api_key_env
(default JUDGE_API_KEY). If no key is set, judging is skipped gracefully.
"""
from __future__ import annotations
import os
import json

RUBRIC = """You are grading an assistant's reply to a user message.
Score it from 1 (useless/rude/wrong) to 10 (helpful, correct, well-written for the task).

User message:
{q}

Assistant reply:
{a}

Respond with ONLY a JSON object: {{"score": <integer 1-10>, "reason": "<short>"}}"""

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _judge_params(cfg):
    """(base_url, key, model). Prefer the dedicated judge key (Groq etc.); if none,
    fall back to the user's Gemini teacher so generation runs still get scored
    with zero extra setup."""
    key = os.environ.get(cfg.judge_api_key_env)
    if key:
        return cfg.judge_base_url, key, cfg.judge_model
    gkey = os.environ.get("GEMINI_API_KEY")
    if gkey:
        from server import settings
        return GEMINI_BASE, gkey, settings.current_gemini_model()
    return None, None, None


def available(cfg) -> bool:
    return _judge_params(cfg)[1] is not None


def score_reply(cfg, question: str, answer: str, _retried: bool = False):
    """Return a float score 1-10, or None if judging is unavailable/failed."""
    base, key, model = _judge_params(cfg)
    if not key:
        return None
    from openai import OpenAI
    client = OpenAI(base_url=base, api_key=key)
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "user", "content": RUBRIC.format(q=question, a=answer)}],
        )
        txt = resp.choices[0].message.content
        obj = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        return float(obj["score"])
    except Exception as e:
        if "429" in str(e) and not _retried:      # brief rate-limit: wait once, retry once
            import time
            time.sleep(6)
            return score_reply(cfg, question, answer, _retried=True)
        print(f"[judge] error: {e}")
        return None
