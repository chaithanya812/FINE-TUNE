# WORKBOOK — how this codebase works (and what to learn from it)

This is the guided tour. Read it top-to-bottom once, then use it as a map while you
read the real files. Every concept an interviewer might ask about is tagged
**[LEARN]** with the exact file where you can see it implemented.

---

## 1. The 30-second mental model

```
you describe a model in chat
        │
   server/agent.py          Gemini "teacher" agent — plans, calls tools
        │
   workspaces/<project>/    project.json + datasets/*.jsonl   (server/store.py)
        │
   finetune_studio/…        the ML engine (data → train → evaluate)
        │
   runs/<run_name>/         adapter/ + report.json  ← the before/after proof
        │
   webui/                   Next.js chat UI — cards, live progress, editor
```

Two ideas hold the whole thing together:

1. **Every dataset row is `{input, target}`** — no matter where it came from
   (generated, uploaded CSV, edited by hand). One shape, every task type.
2. **Every training run must prove itself** — base model vs. fine-tuned model on
   the *same held-out data* the model never saw. No vibes, numbers.

---

## 2. Directory map

| Path | What lives there |
|---|---|
| `config.py` | every knob in one dataclass + `local_4gb` / `colab_t4` presets |
| `finetune_studio/` | the ML engine (below) |
| `server/` | FastAPI backend: agent, jobs, storage, settings |
| `webui/` | Next.js 16 + React 19 chat UI (primary) |
| `web/` | legacy single-file UI (served at `:8000/`) |
| `scripts/` | headless runners (`run_local.py`, `run_autoloop.py`, `make_colab.py`) |
| `workspaces/` | per-project data: `project.json`, `datasets/*.jsonl` (gitignored) |
| `runs/` | training outputs: `adapter/`, `report.json`, checkpoints (gitignored) |

---

## 3. The ML engine — `finetune_studio/`

### `data.py` — normalize everything to `{input, target}`
Loads from three sources (built-in Bitext support set, a CSV, or a workspace
dataset) and returns a `DataBundle(train_df, eval_df, labels, task, …)`.
**[LEARN] train/eval split discipline:** the eval rows are carved off *before*
training and never trained on — that's what makes the before/after numbers honest.

### `tasks.py` — the task-type registry
4 task types (classification / generation / tone / extraction). Each knows its
system prompt, how to turn `{input, target}` into chat messages, and which metric
proves it worked. **[LEARN] prompt engineering as data:** the *training* example is
`[system, user, assistant]`; the *inference* prompt is the same minus the answer.

### `trainer.py` — the actual fine-tuning
QLoRA: load the base model in **4-bit NF4** (bitsandbytes), freeze it, and train
small **LoRA adapter** matrices on top (r=16, α=32) — that's why a 0.5B model
trains in ~2.3 GB of VRAM. Emits progress events (step, loss, ETA, live VRAM),
supports cancel and checkpoint/resume.
**[LEARN — the #1 interview topic]:**
- *LoRA:* instead of updating W (huge), learn ΔW = B·A where A,B are tiny low-rank
  matrices. Train ~1% of parameters.
- *QLoRA:* same, but the frozen base is quantized to 4-bit so it fits in VRAM.
- *Prompt masking:* the loss is only computed on the **assistant's answer tokens**
  — prompt tokens get label `-100` so the model learns to *answer*, not to
  *recite the question*. (Same trick visible in `colab_export.py`.)

### `evaluate.py` — "did it get better?"
Loads the base model ONCE, attaches the adapter, scores **tuned** (adapter on) then
**base** (`disable_adapter()`) on the same held-out set — no second model copy, so
it fits in 4 GB. Classification → accuracy + macro-F1 + per-class table + real
example predictions. Generation → LLM-judge scores.
**[LEARN] `disable_adapter()`** is the elegant bit: one model in memory, two
behaviours.

### `judge.py` — LLM-as-judge
Open-ended replies have no exact answer, so a strong model grades each reply 1-10
against a rubric. Uses a dedicated judge key if set, otherwise **falls back to the
Gemini teacher key** so scoring always works. **[LEARN] LLM-as-judge** is the
standard technique for evaluating generation tasks.

### `synth.py` — synthetic training data (teacher → student distillation)
Asks Gemini for batches of diverse `{input, target}` examples for your task, with
dedup + quality filtering + a **truncation-tolerant JSON parser** (thinking models
sometimes cut the array short — `_salvage()` recovers the complete objects).
**[LEARN] distillation:** a big model writes the lesson, a small model learns it.

### `autotest.py` — adversarial auto-testing
Generates ~10 *hard* edge cases (ambiguous, angry, misspelled), runs them through
your tuned model, scores them (extraction is scored by **real JSON field-matching**,
not vibes) and returns a verdict.

### `dataio.py` — upload parsing + dataset hygiene
CSV/JSONL parsing, column mapping, dedup, class-balance stats, train/val/test
split counts. Powers the spreadsheet editor's save path.

### `catalog.py` / `hyperparam.py` / `autoloop.py`
Model catalog with a conservative **VRAM estimator** (what fits in 4 GB); LR
scaling rules; an auto-improve loop that escalates the recipe until a target
metric is hit.

### `colab_export.py` — the "bigger model" escape hatch
Generates a self-contained **Unsloth** notebook with your project's dataset
embedded — Qwen-3B/7B trains on a free Colab T4. **[LEARN] Unsloth** = the
standard OSS answer to "fine-tune big models free".

### `deploy.py` — export
Adapter zip download, GGUF-conversion script, LM Studio steps, inference snippet.

---

## 4. The server — `server/`

### `app.py` — the API (~30 endpoints)
FastAPI. The interesting groups: `/api/projects*` (CRUD + data upload/generate/
edit + colab), `/api/train` + `WS /ws/{job_id}` (live training events),
`/api/settings` + `/api/gemini/models` (admin), `/api/agent` (the chat).

### `agent.py` — the Gemini agent
**[LEARN] the tool-calling loop** — this is how every AI agent works:
```
messages → model → tool_calls? → run tool → append result → model again → reply
```
8 tools (create_project, propose_plan, generate_training_data, start_training,
get_training_status, try_model, auto_test, get_deploy_info). The system prompt
forces honest behaviour: clarify *route vs answer* before building, never invent
UI, collect enough data (250-400), admit what small models can't do.

### `jobs.py` — background training jobs
One job at a time (one GPU), events buffered for the WebSocket, cancel/resume,
unloads any cached inference model before training (4 GB discipline), records
finished runs into the project.

### `model_service.py` — inference cache
Loads base+adapter once, toggles the adapter for before/after chat, `unload()`
frees VRAM before training.

### `store.py` / `settings.py` / `system.py`
File-based persistence (no DB — every project is a folder with JSON); app
settings (which Gemini model is the teacher — switchable live in `/admin`);
GPU probe + model recommendation.

---

## 5. The frontend — `webui/`

- `src/lib/api.ts` — typed client for every endpoint.
- `src/app/page.tsx` — the whole chat experience: template gallery ("what kind of
  model do you want?"), message stream with a **safe hand-rolled Markdown
  renderer**, plan/data/score/deploy cards, live training panel (WebSocket:
  progress bar, loss, ETA, VRAM), the **spreadsheet-style dataset editor**
  (view/edit/add/delete every row + CSV import/export), Colab scale-up buttons.
- `src/app/admin/page.tsx` — teacher-model picker (live model list from your key).

**[LEARN] the WebSocket pattern:** the backend buffers training events; the UI
subscribes per-job and renders progress live. Note the React StrictMode guard —
dev double-mount must not read as a training failure.

---

## 6. Trace one full run (do this once, in the debugger of your head)

1. Chat: "build a support router" → `POST /api/agent` → Gemini calls
   `create_project` → `propose_plan` (plan card) → you approve.
2. `generate_training_data` → `synth.generate()` batches Gemini calls → rows
   dedup'd → `store.save_dataset()` → data card (you can open the editor, fix
   rows, import CSV — `PUT /api/projects/{pid}/data/{did}`).
3. `start_training` → `jobs.py` starts a thread → `data.load_data()` →
   `trainer` (QLoRA, progress events over `WS /ws/{job_id}`).
4. Training ends → `evaluate.py` scores base vs tuned on held-out rows →
   `runs/<name>/report.json` → run recorded into the project.
5. UI renders before/after + real examples + per-class bars; `auto_test` throws
   edge cases at it; `deploy.py` exports; Colab button for the bigger version.

---

## 7. Interview cheat-sheet (the claims you can defend)

- "I fine-tuned Qwen2.5-0.5B with **QLoRA on a 4 GB laptop GPU**, raising held-out
  intent-classification accuracy **22% → 89%**" — `runs/*/report.json` is the receipt.
- "I built the **full loop**: synthetic data generation with a teacher model,
  human-editable datasets, training with live telemetry, honest base-vs-tuned
  eval, LLM-judge for generation, adversarial auto-test, and export."
- "I know **why** it fits in 4 GB" — 4-bit NF4 base + LoRA adapters + batch 1 +
  gradient accumulation + gradient checkpointing.
- "Fine-tuning teaches **style/format/behaviour**, not facts — for facts you add
  RAG. A 0.5B specialist can beat a frontier model on one narrow task; it will
  never beat it in general."
