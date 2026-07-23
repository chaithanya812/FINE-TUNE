"""The training engine - LoRA / QLoRA supervised fine-tuning.

This is the part that "learns": for each example it runs a forward pass, measures
loss ONLY on the answer tokens (the prompt is masked out with -100), backpropagates,
and nudges the LoRA adapter weights. We use HuggingFace Trainer directly (stable API)
instead of a higher-level wrapper so the mechanics stay visible and version-proof.
"""
from __future__ import annotations
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

from .data import build_messages

# Qwen2.5 linear layers we attach LoRA adapters to.
QWEN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def compute_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16          # Ampere (RTX 30xx) supports bf16 - more stable
    return torch.float16


def load_tokenizer(cfg):
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def build_model(cfg, for_training: bool = True):
    import transformers
    dtype = compute_dtype()
    # transformers >=5 renamed the `torch_dtype` arg to `dtype`.
    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    kwargs = {dtype_key: dtype, "device_map": "auto"}
    if cfg.mode == "qlora":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
    model.config.use_cache = False
    if for_training and cfg.mode == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


def _tokenize(messages, tokenizer, max_len):
    """Tokenize one example and mask the prompt so loss is only on the answer."""
    # Render to text first, then tokenize to plain int lists. Robust across
    # transformers versions - apply_chat_template(tokenize=True) can return an
    # Encoding object that datasets/Arrow cannot serialize.
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer(full_text, add_special_tokens=False)["input_ids"][:max_len]
    prompt = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    labels = list(full)
    for i in range(min(len(prompt), len(full))):
        labels[i] = -100                       # -100 = "don't compute loss here"
    return {"input_ids": full, "labels": labels, "attention_mask": [1] * len(full)}


def _make_dataset(bundle, tokenizer, cfg) -> Dataset:
    rows = [
        _tokenize(build_messages(r, bundle.task, bundle.label_field, bundle.labels), tokenizer, cfg.max_seq_len)
        for _, r in bundle.train_df.iterrows()
    ]
    return Dataset.from_list(rows)


class _Collator:
    """Pads a batch of variable-length examples (input_ids with pad, labels with -100)."""
    def __init__(self, tokenizer):
        self.pad = tokenizer.pad_token_id

    def __call__(self, feats):
        maxlen = max(len(f["input_ids"]) for f in feats)
        ids, lbls, mask = [], [], []
        for f in feats:
            n = maxlen - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad] * n)
            lbls.append(f["labels"] + [-100] * n)
            mask.append(f["attention_mask"] + [0] * n)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(lbls),
            "attention_mask": torch.tensor(mask),
        }


class _ProgressCallback(TrainerCallback):
    """Streams {step, max_steps, loss, epoch, eta_sec, vram} to the UI callback."""
    def __init__(self, cb=None):
        self.cb = cb
        self.t0 = None

    def on_train_begin(self, args, state, control, **kw):
        import time
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kw):
        if self.cb and logs and "loss" in logs:
            import time
            elapsed = (time.time() - self.t0) if self.t0 else 0.0
            step = max(state.global_step, 1)
            eta = int((state.max_steps - state.global_step) * (elapsed / step)) if state.max_steps else None
            vram = None
            try:
                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info()
                    vram = {"used_gb": round((total - free) / 1e9, 2), "total_gb": round(total / 1e9, 2)}
            except Exception:
                pass
            self.cb({
                "step": state.global_step,
                "max_steps": state.max_steps,
                "loss": logs.get("loss"),
                "epoch": round(logs.get("epoch", 0), 2),
                "eta_sec": eta,
                "vram": vram,
            })


class TrainingCancelled(Exception):
    """Raised when the user stops an in-progress run (so callers can clean up VRAM)."""


class _CancelCallback(TrainerCallback):
    """Checks a threading.Event after each step; if set, asks Trainer to stop cleanly."""
    def __init__(self, cancel_event):
        self.cancel_event = cancel_event

    def on_step_end(self, args, state, control, **kw):
        if self.cancel_event is not None and self.cancel_event.is_set():
            control.should_training_stop = True
        return control


def train(cfg, bundle, progress_cb=None, cancel_event=None, resume: bool = False) -> str:
    """Fine-tune and save a LoRA adapter. Returns the adapter directory path.

    cancel_event: a threading.Event; when set, training halts at the next step
                  boundary and TrainingCancelled is raised (no adapter is saved).
    resume:       continue from the latest checkpoint in the run's checkpoints/ dir.
    """
    tokenizer = load_tokenizer(cfg)
    model = build_model(cfg, for_training=True)

    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=QWEN_TARGETS,
    ))
    model.print_trainable_parameters()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()   # required for gradient checkpointing + PEFT

    ds = _make_dataset(bundle, tokenizer, cfg)
    dtype_bf16 = compute_dtype() == torch.bfloat16

    ckpt_root = cfg.output_dir / "checkpoints"
    if not resume and ckpt_root.exists():
        import shutil
        shutil.rmtree(ckpt_root, ignore_errors=True)   # fresh run: never resume stale state

    args = TrainingArguments(
        output_dir=str(ckpt_root),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",                # periodic checkpoints -> resume after a crash/shutdown
        save_steps=cfg.save_steps,
        save_total_limit=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",             # bitsandbytes paged optimizer - saves VRAM
        bf16=dtype_bf16,
        fp16=not dtype_bf16,
        report_to=[],
        seed=cfg.seed,
    )

    callbacks = [_ProgressCallback(progress_cb)]
    if cancel_event is not None:
        callbacks.append(_CancelCallback(cancel_event))

    trainer = Trainer(
        model=model, args=args, train_dataset=ds,
        data_collator=_Collator(tokenizer),
        callbacks=callbacks,
    )

    resume_ckpt = True if (resume and ckpt_root.exists() and any(ckpt_root.glob("checkpoint-*"))) else None
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # If the user cancelled, don't save a half-trained adapter as the final model.
    cancelled = cancel_event is not None and cancel_event.is_set()
    if not cancelled:
        model.save_pretrained(str(cfg.adapter_dir))
        tokenizer.save_pretrained(str(cfg.adapter_dir))

    # Free VRAM before evaluation loads its own copy (critical on a 4GB GPU).
    del trainer, model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if cancelled:
        raise TrainingCancelled()
    return str(cfg.adapter_dir)
