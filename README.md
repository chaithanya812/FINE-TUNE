# Fine-Tune Studio

**A local "ChatGPT for fine-tuning."** You describe the model you want in plain
English — an AI agent plans it, generates (or imports) editable training data,
fine-tunes a small open model with **QLoRA on a 4 GB laptop GPU**, and then *proves*
the improvement on held-out data it never trained on.

> Measured on this machine: intent-classification accuracy **22% → 89%** (base
> Qwen2.5-0.5B vs fine-tuned, same held-out set) in a ~15-minute run on an
> RTX 3050 Laptop (4 GB). The receipt for every run is written to
> `runs/<name>/report.json`.

Built with **PyTorch · HuggingFace (transformers / peft / bitsandbytes) · FastAPI ·
Next.js 16 / React 19 · Gemini (teacher agent) · Unsloth (Colab export)**.

---

## How it works

```
 "Build a support bot that sorts messages by intent"
        │
   Gemini agent ── clarifies ROUTE vs ANSWER, proposes a plan you approve
        │
   data ── teacher-generated {input, target} rows → spreadsheet editor
        │         (view/edit/add/delete every row, CSV import/export)
        │
   train ── QLoRA 4-bit on Qwen2.5-0.5B, live progress (loss · ETA · VRAM)
        │
   evaluate ── base vs tuned on held-out data
        │         classification → accuracy + macro-F1 + per-class
        │         generation     → LLM-as-judge (auto-falls-back to Gemini)
        │         extraction     → real JSON field matching
        │
   prove ── before/after card + real example predictions + adversarial auto-test
        │
   ship ── adapter download · LM Studio steps · GGUF script · or a one-click
            Unsloth Colab notebook (your data embedded) for Qwen-3B/7B on a free T4
```

## Features

- **Chat-first UX** — a Gemini-powered agent runs the whole flow through
  conversation, with typed artifact cards (plan / data / score / deploy).
- **Use-case templates** — Router/Classifier, Assistant-in-your-voice, Data
  extractor, Tone rewriter, Domain reasoner — each with honest notes about what
  fine-tuning can and can't do for that shape of problem.
- **Editable datasets** — every generated example is visible and editable in a
  spreadsheet-style editor (plus CSV import/export). Your data, not a black box.
- **Honest evaluation** — the same held-out examples through the base and tuned
  model; real per-class breakdowns; wins *and* regressions shown.
- **4 GB discipline** — 4-bit NF4 base + LoRA adapters + gradient checkpointing;
  one cached model with `disable_adapter()` for before/after; VRAM telemetry live.
- **Teacher-model admin** — `/admin` lists the Gemini models your key can call and
  switches the teacher live (no restart).
- **Scale-up path** — generates a self-contained Unsloth Colab notebook with your
  dataset baked in; checkpoint/resume; cancel; auto-improve loop.

## Run it

```powershell
# deps (once): torch with CUDA, then the rest
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# backend (FastAPI + agent) on :8000       — needs GEMINI_API_KEY in .env
.\.venv\Scripts\python.exe -m uvicorn server.app:app --port 8000
# frontend (Next.js) on :3000
pnpm -C webui dev
```

Headless engine proof (no UI): `.\.venv\Scripts\python.exe scripts\run_local.py`

## Repo map

See **[WORKBOOK.md](WORKBOOK.md)** for the full guided tour. Short version:

| Path | Role |
|---|---|
| `finetune_studio/` | ML engine: data → tasks → trainer (QLoRA) → evaluate → judge · synth · autotest · deploy · colab_export · templates |
| `server/` | FastAPI: agent (tool-calling loop) · jobs · store · settings · model service |
| `webui/` | Next.js chat UI: template gallery, dataset editor, live training panel, admin |
| `config.py` | every knob + `local_4gb` / `colab_t4` presets |
| `runs/` · `workspaces/` | training outputs · per-project data (both gitignored) |

## Honest limits (by design, stated in-product)

- Fine-tuning teaches **style / format / behaviour — not facts**. A 0.5B specialist
  can beat a frontier model on one narrow task; it won't be a general assistant.
  (Knowledge → RAG, on the roadmap.)
- The 4 GB local path is capped at 0.5B; bigger models go through the Colab path.
- Cloud-GPU one-click training is a stub ("coming soon") — notebook export works today.
