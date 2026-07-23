"""Central configuration for the fine-tuning studio.

Everything you can tune lives here. Two presets are provided:
  - "local_4gb": Qwen2.5-0.5B + QLoRA, sized to fit a 4GB laptop GPU (RTX 3050).
  - "colab_t4" : Qwen2.5-1.5B, bigger batches, for a free Colab T4 (16GB).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    # --- what to train ------------------------------------------------------
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    mode: str = "qlora"            # "qlora" (4-bit, fits 4GB) | "lora" (16-bit)
    task: str = "classification"   # "classification" (accuracy) | "generation" (LLM-judge)

    # --- data ---------------------------------------------------------------
    dataset: str = "bitext"        # "bitext" (auto-download) | "csv" | "workspace"
    csv_path: str = str(PROJECT_ROOT / "data" / "sample_support.csv")
    project_id: str | None = None  # workspace dataset source (dataset="workspace")
    dataset_id: str | None = None
    label_field: str = "intent"    # classification target: "intent" (27-way) | "category" (11-way)
    n_train: int = 2000
    n_eval: int = 400
    max_seq_len: int = 512

    # --- LoRA hyper-parameters ---------------------------------------------
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # --- optimization -------------------------------------------------------
    epochs: float = 3.0
    lr: float = 2e-4
    batch_size: int = 1
    grad_accum: int = 8            # effective batch = batch_size * grad_accum
    warmup_ratio: float = 0.03
    seed: int = 42
    save_steps: int = 50           # periodic checkpoint cadence (enables crash/shutdown resume)

    # --- output -------------------------------------------------------------
    run_name: str = "run1"
    output_root: str = str(PROJECT_ROOT / "runs")

    # --- LLM-as-judge (generation task only; OpenAI-compatible endpoint) ----
    judge_base_url: str = "https://api.groq.com/openai/v1"
    judge_model: str = "llama-3.3-70b-versatile"
    judge_api_key_env: str = "JUDGE_API_KEY"

    # ------------------------------------------------------------------ paths
    @property
    def output_dir(self) -> Path:
        p = Path(self.output_root) / self.run_name
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def adapter_dir(self) -> Path:
        return self.output_dir / "adapter"

    # --------------------------------------------------------------- presets
    @classmethod
    def preset(cls, name: str, **overrides) -> "Config":
        if name == "local_4gb":
            base = dict(
                model_id="Qwen/Qwen2.5-0.5B-Instruct", mode="qlora",
                n_train=2000, n_eval=400, max_seq_len=512,
                batch_size=1, grad_accum=8,
            )
        elif name == "colab_t4":
            base = dict(
                model_id="Qwen/Qwen2.5-1.5B-Instruct", mode="qlora",
                n_train=5000, n_eval=600, max_seq_len=768,
                batch_size=4, grad_accum=4,
            )
        else:
            raise ValueError(f"unknown preset {name!r} (use 'local_4gb' or 'colab_t4')")
        base.update(overrides)
        return cls(**base)

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path or (self.output_dir / "config.json"))
        path.write_text(json.dumps(asdict(self), indent=2))
        return path
