"""Data loading + prompt building.

Everything is normalized to two columns — **input** and **target** — no matter where
it came from (the built-in Bitext support set, a user CSV, or a workspace dataset the
user uploaded / generated). The task registry (tasks.py) turns {input, target} into
the right chat messages, so this module is task-agnostic.
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from . import tasks

BITEXT_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


@dataclass
class DataBundle:
    train_df: pd.DataFrame        # columns: input, target
    eval_df: pd.DataFrame
    labels: list[str]
    task: str
    label_field: str              # always "target" now (kept for call-site compatibility)
    eval_set_id: str | None = None  # frozen exam this bundle scores on (Phase 2), if any


# --------------------------------------------------------------- source loaders
def _load_workspace(cfg) -> pd.DataFrame:
    """A dataset the user uploaded or generated, stored as {input, target} jsonl."""
    from server import store
    rows = store.get_dataset_rows(cfg.project_id, cfg.dataset_id)
    if not rows:
        raise ValueError(f"workspace dataset {cfg.dataset_id!r} is empty or missing")
    df = pd.DataFrame(rows)
    if "input" not in df.columns or "target" not in df.columns:
        raise ValueError("workspace dataset rows must have 'input' and 'target' fields")
    return df[["input", "target"]].astype(str)


def _load_legacy(cfg) -> pd.DataFrame:
    """Bitext (auto-download) or a user CSV → normalized to input/target."""
    if cfg.dataset == "csv":
        df = pd.read_csv(cfg.csv_path)
    else:
        try:
            from datasets import load_dataset
            df = load_dataset(BITEXT_ID, split="train").to_pandas()
        except Exception as e:  # offline / no HF access -> bundled sample
            print(f"[data] could not load Bitext ({e}); using bundled sample CSV")
            df = pd.read_csv(cfg.csv_path)

    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    inst = pick("instruction", "utterance", "text", "message", "input")
    target_col = (cfg.label_field if cfg.task == "classification"
                  else pick("response", "answer", "output", "target"))
    if inst is None or target_col is None or target_col not in df.columns:
        raise ValueError(f"CSV must have an input column and a '{cfg.label_field}'/response column")
    out = pd.DataFrame({
        "input": df[inst].astype(str),
        "target": df[target_col].astype(str),
    })
    return out


def load_data(cfg) -> DataBundle:
    df = _load_workspace(cfg) if cfg.dataset == "workspace" else _load_legacy(cfg)
    df = df[df["input"].astype(str).str.len() > 0]
    df = df[df["target"].astype(str).str.len() > 0].reset_index(drop=True)

    labels = sorted(df["target"].unique().tolist()) if cfg.task == "classification" else []

    # Phase 2 — frozen eval set: a workspace project scores every version on ONE
    # immutable, leakage-guarded exam so numbers are comparable across runs. The
    # training rows are whatever's left after holding that exam out (even as the
    # dataset is edited later). Guarded: any failure falls back to the random split.
    if cfg.dataset == "workspace" and cfg.project_id and cfg.dataset_id:
        try:
            from finetune_studio import evalsets
            from server import db, store
            raw_rows = store.get_dataset_rows(cfg.project_id, cfg.dataset_id)  # keeps 'golden'
            es = evalsets.get_or_create_frozen(
                cfg.project_id, raw_rows, cfg.task,
                source_dataset_version_id=cfg.dataset_id, seed=cfg.seed)
            eval_rows = db.get_eval_set_rows(es["id"])
            train_rows = evalsets.held_out_train(raw_rows, eval_rows)
            if eval_rows and train_rows:
                eval_df = pd.DataFrame(eval_rows)[["input", "target"]].astype(str).reset_index(drop=True)
                tdf = pd.DataFrame(train_rows)[["input", "target"]].astype(str)
                tdf = tdf.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
                train_df = tdf.iloc[: cfg.n_train].reset_index(drop=True)
                return DataBundle(train_df, eval_df, labels, cfg.task, "target",
                                  eval_set_id=es["id"])
        except Exception as e:
            print(f"[data] frozen eval unavailable ({e}); falling back to random split")

    # Legacy path (bitext/csv, or if freezing was not possible): random split.
    df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    df = df.iloc[: min(len(df), cfg.n_train + cfg.n_eval)]
    eval_df = df.iloc[: cfg.n_eval].reset_index(drop=True)
    train_df = df.iloc[cfg.n_eval:].reset_index(drop=True)
    return DataBundle(train_df, eval_df, labels, cfg.task, "target")


# ------------------------------------------------ message building (task-aware)
def build_messages(row, task, label_field, labels) -> list[dict]:
    """Full chat (system, user, assistant) for TRAINING. `label_field` is 'target'."""
    return tasks.build_train_messages(task, {"input": row["input"], "target": row[label_field]}, labels)


def build_prompt_messages(instruction, task, labels) -> list[dict]:
    """Prompt only (system, user) for INFERENCE / EVAL."""
    return tasks.build_prompt_messages(task, instruction, labels)
