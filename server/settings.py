"""App settings — currently just which Gemini model is the "teacher".

File-based like server/store.py (single-user local app, no DB). The chosen model
is read *dynamically* on every teacher call (synth / autotest / agent), so changing
it in the admin page takes effect immediately — no server restart needed.

Precedence for the active model:
    settings.json  >  env GEMINI_MODEL  >  DEFAULT_MODEL
"""
from __future__ import annotations
import json
import os
import threading
import urllib.request
from pathlib import Path

from config import PROJECT_ROOT

_FILE = PROJECT_ROOT / "settings.json"
_LOCK = threading.RLock()

# gemini-2.5-flash has a tiny free-tier cap (20 requests/day) that users hit fast.
# The lite models are faster and have far more free-tier headroom for teacher work.
DEFAULT_MODEL = "gemini-3.1-flash-lite"

# Shown first in the admin picker — good, cheap, high-free-tier text models for the
# teacher/agent role. (The full live list is fetched from the API; this is just the
# "recommended" flag + the offline fallback if the list call fails.)
RECOMMENDED = [
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
]

# Substrings that mark a model as NOT a plain text/chat teacher (image, speech, etc.).
_NON_TEXT = ("image", "tts", "embedding", "aqa", "robotics", "computer-use",
             "lyria", "nano-banana", "veo", "imagen", "audio")

_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200&key="


def _read() -> dict:
    if _FILE.exists():
        try:
            return json.loads(_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _write(d: dict) -> None:
    _FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")


def current_gemini_model() -> str:
    """The active teacher model. Read on every call so admin changes apply live."""
    with _LOCK:
        chosen = _read().get("gemini_model")
    return (chosen or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL).strip()


def get_settings() -> dict:
    return {
        "gemini_model": current_gemini_model(),
        "default_model": DEFAULT_MODEL,
        "key_set": bool(os.environ.get("GEMINI_API_KEY")),
    }


def update_settings(gemini_model: str | None = None) -> dict:
    with _LOCK:
        s = _read()
        if gemini_model and gemini_model.strip():
            s["gemini_model"] = gemini_model.strip()
        _write(s)
    return get_settings()


def _fallback_models() -> list[dict]:
    return [{"id": m, "label": m, "description": "", "recommended": m in RECOMMENDED}
            for m in RECOMMENDED]


def list_gemini_models() -> dict:
    """Live list of text models the user's key can call (for the admin dropdown).

    Filters to models supporting generateContent and drops non-text ones (image,
    tts, embeddings...). Falls back to the curated RECOMMENDED list on any error.
    """
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return {"models": _fallback_models(), "source": "fallback",
                "error": "No GEMINI_API_KEY set — add it to .env."}
    try:
        with urllib.request.urlopen(_MODELS_URL + key, timeout=15) as r:
            data = json.load(r)
    except Exception as e:
        return {"models": _fallback_models(), "source": "fallback", "error": str(e)}

    models: list[dict] = []
    for m in data.get("models", []):
        if "generateContent" not in m.get("supportedGenerationMethods", []):
            continue
        mid = m.get("name", "").replace("models/", "")
        if not mid or any(x in mid.lower() for x in _NON_TEXT):
            continue
        models.append({
            "id": mid,
            "label": m.get("displayName") or mid,
            "description": (m.get("description") or "").strip()[:160],
            "recommended": mid in RECOMMENDED,
        })
    # recommended first (in RECOMMENDED order), then the rest alphabetically
    rank = {m: i for i, m in enumerate(RECOMMENDED)}
    models.sort(key=lambda x: (rank.get(x["id"], len(RECOMMENDED)), x["id"]))
    if not models:
        return {"models": _fallback_models(), "source": "fallback",
                "error": "No text models returned."}
    return {"models": models, "source": "live"}
