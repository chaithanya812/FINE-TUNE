"""Generate a self-contained Google Colab notebook (.ipynb) — the "no GPU / bigger
model" path.

A 4 GB laptop can't train a 3-4B model, so we hand the user a notebook that runs the
SAME idea on a free Colab T4 using **Unsloth** (2x faster, ~70% less VRAM → Qwen 3-4B
fits free). The user's own generated/edited dataset is embedded directly in the
notebook, so it's genuinely one-click: open → Runtime T4 → Run all → download adapter.
No dependency on this repo.
"""
from __future__ import annotations
import json
from pathlib import Path

from finetune_studio import tasks

INSTALL = r'''# Unsloth = fast, low-VRAM fine-tuning. This is all you need on a fresh Colab.
!pip install -q unsloth
# keep deps fresh (Colab pins can lag):
!pip install -q --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" || true'''

PIPELINE = r'''
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
import torch, json

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name   = MODEL_ID,
    max_seq_length = MAX_SEQ_LEN,
    load_in_4bit = True,       # QLoRA
    dtype        = None,       # auto (bf16 on T4-newer / fp16 otherwise)
)

model = FastLanguageModel.get_peft_model(
    model, r = 16, lora_alpha = 16, lora_dropout = 0, bias = "none",
    target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing = "unsloth", random_state = 42,
)

# ---- your data (embedded from Fine-Tune Studio) -> chat text -----------------
def to_text(row, with_answer=True):
    msgs = [{"role":"system","content":SYSTEM},
            {"role":"user","content":str(row["input"])}]
    if with_answer:
        msgs.append({"role":"assistant","content":str(row["target"])})
    return tokenizer.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=not with_answer)

train_rows = DATA[N_EVAL:]
eval_rows  = DATA[:N_EVAL]
ds = Dataset.from_list([{"text": to_text(r, True)} for r in train_rows])
print(f"train={len(train_rows)}  eval={len(eval_rows)}  labels={len(LABELS)}")

trainer = SFTTrainer(
    model = model, tokenizer = tokenizer, train_dataset = ds,
    args = SFTConfig(
        dataset_text_field = "text", max_seq_length = MAX_SEQ_LEN,
        per_device_train_batch_size = 2, gradient_accumulation_steps = 4,
        warmup_steps = 5, num_train_epochs = EPOCHS, learning_rate = 2e-4,
        logging_steps = 5, optim = "adamw_8bit", weight_decay = 0.01,
        lr_scheduler_type = "linear", seed = 42, output_dir = "outputs", report_to = "none",
    ),
)
# train only on the assistant's answer, not the prompt (Qwen chat markers):
try:
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part    = "<|im_start|>assistant\n",
    )
except Exception as e:
    print("train_on_responses_only skipped:", e)

trainer.train()
model.save_pretrained("adapter"); tokenizer.save_pretrained("adapter")
print("saved adapter/")
'''

EVAL = r'''
FastLanguageModel.for_inference(model)

def generate(message, max_new):
    text = tokenizer.apply_chat_template(
        [{"role":"system","content":SYSTEM},{"role":"user","content":str(message)}],
        add_generation_prompt=True, tokenize=False)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def match(t):
    t = t.lower().strip()
    for l in LABELS:
        if t == l.lower(): return l
    for l in LABELS:
        if l.lower() in t: return l
    return "unknown"

if TASK == "classification" and eval_rows:
    def acc():
        ok = 0
        for r in eval_rows:
            if match(generate(r["input"], 16)) == str(r["target"]): ok += 1
        return ok / len(eval_rows)
    after = acc()
    with model.disable_adapter():
        before = acc()
    print(f"Base (before):       {before*100:5.1f}%")
    print(f"Fine-tuned (after):  {after*100:5.1f}%")
    print(f"Improvement:         +{(after-before)*100:.1f} points on {len(eval_rows)} held-out examples")
else:
    print("Sample replies from your fine-tuned model:")
    for r in eval_rows[:3]:
        print("Q:", r["input"]); print("A:", generate(r["input"], 120)); print("-"*40)
'''

DOWNLOAD = r'''
import shutil
shutil.make_archive("adapter", "zip", "adapter")
print("adapter.zip ready.")
try:
    from google.colab import files
    files.download("adapter.zip")   # pull the trained LoRA adapter to your machine
except Exception:
    print("Find adapter.zip in the file browser on the left.")

# OPTIONAL — export a GGUF you can run in LM Studio / Ollama (uncomment):
# model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method="q4_k_m")

# OPTIONAL — push to your Hugging Face account (uncomment + set token):
# model.push_to_hub("your-username/my-finetune", token="hf_...")
'''


def _md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def _code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")}


def build_project_notebook(project: dict, rows: list[dict], labels: list[str] | None = None,
                           colab_model: str | None = None) -> dict:
    """A one-click Colab notebook that trains `project` on its own `rows` via Unsloth."""
    task = project.get("task_type", "classification")
    labels = labels or (sorted({str(r["target"]) for r in rows}) if task == "classification" else [])
    system = tasks._system(task, labels)
    model_id = colab_model or "Qwen/Qwen2.5-3B-Instruct"
    n = len(rows)
    n_eval = max(4, min(60, int(n * 0.15))) if n else 0

    config_cell = (
        "# ==== EDIT THESE IF YOU WANT ====\n"
        f'MODEL_ID    = "{model_id}"   # try "Qwen/Qwen2.5-7B-Instruct" too — Unsloth fits it on a free T4\n'
        f'TASK        = "{task}"\n'
        f"MAX_SEQ_LEN = 1024\n"
        f"EPOCHS      = 3\n"
        f"N_EVAL      = {n_eval}\n\n"
        "SYSTEM = " + json.dumps(system) + "\n"
        "LABELS = " + json.dumps(labels) + "\n\n"
        "# Your dataset, exported straight from Fine-Tune Studio:\n"
        "DATA = " + json.dumps(rows, ensure_ascii=False) + "\n"
    )

    cells = [
        _md(f"# {project.get('name','Your model')} — Colab trainer (Unsloth)\n"
            f"**Goal:** {project.get('goal','')}\n\n"
            "Your laptop's 4 GB GPU can't fit a 3-4B model, so this trains it on Colab's **free T4** "
            "using [Unsloth](https://github.com/unslothai/unsloth) — same idea, bigger model.\n\n"
            "**How to run:** `Runtime → Change runtime type → T4 GPU`, then `Runtime → Run all`. "
            "At the end it downloads your trained adapter. Your data is already baked in below."),
        _md("### 1 · Install Unsloth"),
        _code(INSTALL),
        _md("### 2 · Configure (your data is embedded here)"),
        _code(config_cell),
        _md("### 3 · Load the model & train (QLoRA via Unsloth)"),
        _code(PIPELINE),
        _md("### 4 · Did it get better? (base vs fine-tuned on held-out data)"),
        _code(EVAL),
        _md("### 5 · Download your trained adapter"),
        _code(DOWNLOAD),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def write_project_notebook(project: dict, rows: list[dict], labels=None,
                           colab_model=None, path: str | Path | None = None) -> Path:
    nb = build_project_notebook(project, rows, labels, colab_model)
    out = Path(path) if path else Path(f"{project.get('id','project')}_colab.ipynb")
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return out
