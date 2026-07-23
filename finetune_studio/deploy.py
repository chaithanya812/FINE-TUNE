"""Deploy / export helpers (Stage 7).

- describe an adapter for download + how to load it in LM Studio,
- generate a self-contained merge->GGUF script the user can run (we do NOT hard-depend
  on llama.cpp being installed — we hand them a ready script),
- a copy-paste inference API snippet,
- run versioning: compare two runs, delete an old one to free disk.
"""
from __future__ import annotations
import json
import pathlib

from config import Config


def _run_dir(run_name: str) -> pathlib.Path:
    return pathlib.Path(Config().output_root) / run_name


def _dir_size_mb(p: pathlib.Path) -> float:
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return round(total / 1e6, 1)


def _base_model(run_name: str) -> str:
    cfgp = _run_dir(run_name) / "config.json"
    if cfgp.exists():
        return json.loads(cfgp.read_text()).get("model_id", "Qwen/Qwen2.5-0.5B-Instruct")
    return "Qwen/Qwen2.5-0.5B-Instruct"


def merge_gguf_script(run_name: str) -> str:
    base = _base_model(run_name)
    adapter = str((_run_dir(run_name) / "adapter").resolve())
    return f'''# Merge your LoRA adapter into the base model, then convert to GGUF for
# Ollama / LM Studio. Run in the project venv. Needs llama.cpp checked out.
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base}", dtype="auto")
tok = AutoTokenizer.from_pretrained("{base}")
merged = PeftModel.from_pretrained(base, r"{adapter}").merge_and_unload()
merged.save_pretrained("merged_model"); tok.save_pretrained("merged_model")
print("Merged -> ./merged_model")

# Then (in a llama.cpp checkout):
#   python convert_hf_to_gguf.py ./merged_model --outfile model.gguf --outtype q8_0
#   -> load model.gguf in LM Studio (My Models) or `ollama create`.
'''


def inference_snippet(run_name: str) -> str:
    return (
        "# Start the backend (uvicorn server.app:app --port 8000), then:\n"
        "curl -X POST http://localhost:8000/api/chat \\\n"
        '  -H "Content-Type: application/json" \\\n'
        f'  -d \'{{"run_name": "{run_name}", "message": "where is my order?", "use_adapter": true}}\''
    )


def export_info(run_name: str) -> dict | None:
    d = _run_dir(run_name)
    adapter = d / "adapter"
    if not adapter.exists():
        return None
    return {
        "run_name": run_name,
        "base_model": _base_model(run_name),
        "adapter_path": str(adapter.resolve()),
        "size_mb": _dir_size_mb(adapter),
        "files": sorted(f.name for f in adapter.iterdir() if f.is_file()),
        "lm_studio_steps": [
            "The adapter alone can't be loaded directly in LM Studio — merge it first.",
            "Run the merge->GGUF script below to get a single model.gguf.",
            "In LM Studio → My Models → import the .gguf, then chat with it.",
        ],
        "gguf_script": merge_gguf_script(run_name),
        "inference_snippet": inference_snippet(run_name),
    }


def compare(run_a: str, run_b: str) -> dict:
    def rep(name):
        f = _run_dir(name) / "report.json"
        return json.loads(f.read_text()) if f.exists() else None
    return {"a": {"run_name": run_a, "report": rep(run_a)},
            "b": {"run_name": run_b, "report": rep(run_b)}}


def delete_run(run_name: str) -> bool:
    import shutil
    d = _run_dir(run_name)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        return True
    return False


def zip_adapter(run_name: str) -> pathlib.Path | None:
    """Zip the adapter dir to a temp file for download; returns the zip path."""
    import shutil
    import tempfile
    adapter = _run_dir(run_name) / "adapter"
    if not adapter.exists():
        return None
    out = pathlib.Path(tempfile.gettempdir()) / f"{run_name}_adapter"
    zpath = shutil.make_archive(str(out), "zip", str(adapter))
    return pathlib.Path(zpath)
