"""Gemini-powered agent that runs the whole 7-stage flow through chat.

The agent creates a project, proposes a plan the user approves, gets/generates data,
trains, reports results with real examples, auto-tests, and hands off deploy info —
each step surfaced to the UI as a typed "action" artifact (plan card, data card,
scorecard, deploy card). Uses Gemini via its OpenAI-compatible endpoint.
"""
from __future__ import annotations
import os
import json
import pathlib

from config import Config


def _load_env():
    p = pathlib.Path(__file__).resolve().parent.parent / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


def agent_available() -> bool:
    return bool(GEMINI_KEY)


SYSTEM = """You are the assistant inside "Fine-Tune Studio", which lets everyday, non-technical
people create their OWN small AI model just by chatting.

FIRST, THE ONE THING THAT MATTERS MOST — figure out what SHAPE of model they need, because
"a chatbot" means very different things and picking wrong wastes their time:
  • ROUTE messages into buckets (intents, topics, spam) -> task_type "classification". It only
    ever outputs ONE label; it does NOT answer questions.
  • ANSWER / chat in their voice -> "generation". Warn them honestly: fine-tuning teaches STYLE
    and FORMAT, not facts — for it to know their specific docs/prices they'll later want RAG, and
    a genuinely helpful assistant needs a bigger model (the Colab/cloud path), not the 0.5B local one.
  • EXTRACT structured data (messy text -> JSON) -> "extraction".
  • REWRITE text into a tone/style -> "tone".
If their ask is ambiguous ("build me a chatbot"), ASK which of these they mean before creating the
project. Never silently build a classifier for someone who wanted an assistant.

Be honest about limits, always: a small local model will NOT be "as good as Gemini" in general —
the win is that it BEATS the big models on their ONE narrow task while being free, private and fast.
Say this plainly if they expect a general genius.

THE FLOW (tools):
1. PLAN: once the shape is clear, create_project, then propose_plan and show it. Ask to approve.
   Call propose_plan AT MOST ONCE per project — never re-propose unless the plan actually changed.
2. DATA: collect ENOUGH to actually work. For classification, get their category list (3-8 is ideal)
   and aim for ~250-400 examples spread across the classes (never a token 40-60 for a real bot).
   Offer generate_training_data (teacher AI). After generating, point them at the exact UI: the
   "View & edit all N" button ON THE DATA CARD in this chat — it opens a spreadsheet-style editor
   where they can fix any example, delete bad ones, add their own real ones, and import/export CSV.
   (There is no separate "Data tab" — never invent UI that doesn't exist.) Encourage adding a few
   of their OWN real examples so the model sounds like THEM, not like the teacher AI.
3. TRAIN: only after plan approved AND data ready, call start_training. Say it's underway; they see
   live progress — never pretend it finished.
4. RESULTS: on "how did it do?", call get_training_status; ALWAYS show 1-2 real before/after examples.
   If the held-out set is tiny, say the numbers are only a rough read and suggest more data.
5. TEST: offer try_model and auto_test (10 hard cases + verdict).
6. IMPROVE: if a class/topic is weak, offer to generate more data for it and retrain a new version.
7. DEPLOY: when happy, get_deploy_info for download + LM Studio + API. For bigger models, point them
   to the "Open in Colab" button.

Style: warm, plain language, minimal jargon. Pick sensible defaults; guide, don't interrogate.
Celebrate wins simply ("22% → 89% on messages it had never seen!")."""


TOOLS = [
    {"type": "function", "function": {
        "name": "create_project",
        "description": "Start a new project for the user's goal. Call this first when they describe what they want.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Short project name."},
            "goal": {"type": "string", "description": "The user's goal in plain English."},
            "task_type": {"type": "string", "enum": ["classification", "generation", "tone", "extraction"]},
        }, "required": ["name", "goal", "task_type"]}}},
    {"type": "function", "function": {
        "name": "propose_plan",
        "description": "Compute a training plan for the current project (model, method, est. time/cost, #examples) to show the user for approval.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "generate_training_data",
        "description": "Generate synthetic training examples with the teacher AI for the current project. Confirm with the user first.",
        "parameters": {"type": "object", "properties": {
            "n": {"type": "integer", "description": "How many examples. Aim for 250-400 for a real bot; default 250."},
            "labels": {"type": "array", "items": {"type": "string"},
                       "description": "For classification: the label set to spread across."},
        }}}},
    {"type": "function", "function": {
        "name": "start_training",
        "description": "Begin fine-tuning for the current project (uses its dataset, or the built-in support set if none). Returns a job_id.",
        "parameters": {"type": "object", "properties": {
            "epochs": {"type": "integer", "description": "Passes over the data. Default 2."},
            "label_field": {"type": "string", "enum": ["intent", "category"],
                            "description": "Only for the built-in support demo. Default intent."},
        }}}},
    {"type": "function", "function": {
        "name": "get_training_status",
        "description": "Check the current/most-recent run: status, before/after accuracy, and real example predictions to show as proof. job_id optional.",
        "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "try_model",
        "description": "Run one message through the finished tuned model and return its prediction.",
        "parameters": {"type": "object", "properties": {
            "message": {"type": "string"}}, "required": ["message"]}}},
    {"type": "function", "function": {
        "name": "auto_test",
        "description": "Generate ~10 hard edge cases, run them through the tuned model, score them, and return a scorecard + verdict.",
        "parameters": {"type": "object", "properties": {
            "n": {"type": "integer", "description": "How many edge cases. Default 10."}}}}},
    {"type": "function", "function": {
        "name": "get_deploy_info",
        "description": "Get download + LM Studio + inference-API instructions for the finished model.",
        "parameters": {"type": "object", "properties": {}}}},
]


def _client():
    from openai import OpenAI
    return OpenAI(base_url=GEMINI_BASE, api_key=GEMINI_KEY)


# --------------------------------------------------------------- tool handlers
def _proj(state):
    from server import store
    pid = state.get("project_id")
    return store.get_project(pid) if pid else None


def _create_project(state, args):
    from server import store
    from server import system as system_info
    rec = system_info.system_summary().get("recommended_model", "Qwen/Qwen2.5-0.5B-Instruct")
    proj = store.create_project(args.get("name", "My model"), goal=args.get("goal", ""),
                                task_type=args.get("task_type", "classification"), base_model=rec)
    state["project_id"] = proj["id"]
    return ({"project_id": proj["id"], "name": proj["name"], "task_type": proj["task_type"],
             "base_model": proj["base_model"]},
            {"type": "project_created", "project_id": proj["id"], "name": proj["name"]})


def _propose_plan(state, args):
    from server import store, system as system_info
    from finetune_studio import catalog
    proj = _proj(state)
    if not proj:
        return {"error": "no project yet — create one first"}, None
    info = catalog.get(proj["base_model"])
    n_examples = 0
    if proj.get("dataset_id"):
        meta = store.get_dataset_meta(proj["id"], proj["dataset_id"]) or {}
        n_examples = meta.get("n", 0)
    plan = {
        "task_type": proj["task_type"],
        "base_model": proj["base_model"],
        "base_model_label": info.label if info else proj["base_model"],
        "method": "QLoRA (4-bit)",
        "est_train_gb": catalog.estimate_train_gb(info) if info else None,
        "n_examples": n_examples,
        "needs_colab": not (system_info.system_summary()["gpu"].get("free_gb", 0)
                            >= (catalog.estimate_train_gb(info) if info else 99)),
        "est_time_min": "10-20 min on your GPU",
    }
    store.update_project(proj["id"], plan=plan, status="planned")
    return {"plan": plan}, {"type": "plan", "plan": plan, "project_id": proj["id"]}


def _generate_data(state, args):
    from server import store
    from finetune_studio import synth
    proj = _proj(state)
    if not proj:
        return {"error": "no project yet"}, None
    if not synth.available():
        return {"error": "No GEMINI_API_KEY set for data generation"}, None
    n = int(args.get("n", 250))
    labels = args.get("labels")
    out = synth.generate(proj["task_type"], proj.get("goal") or proj["name"], n=n, labels=labels)
    if not out["rows"]:
        return {"error": "generation failed", "detail": out["errors"]}, None
    from finetune_studio import dataio
    rows = dataio.dedup(out["rows"])
    st = dataio.stats(rows, proj["task_type"])
    st["split"] = dataio.split_counts(rows)
    did = store.save_dataset(proj["id"], rows, {"source": "synthetic", "task": proj["task_type"], **st})
    store.update_project(proj["id"], dataset_id=did, status="data_ready")
    return ({"created": len(rows), "n_classes": st.get("n_classes"), "preview": rows[:6]},
            {"type": "data_generated", "stats": st, "preview": rows[:12], "n": len(rows),
             "project_id": proj["id"], "dataset_id": did})


def _start_training(jm, state, args):
    from server import store
    proj = _proj(state)
    if not proj:
        proj = store.create_project("My model", task_type="classification")
        state["project_id"] = proj["id"]
    pid = proj["id"]
    run_name = f"{pid}_v{len(proj.get('runs', [])) + 1}"
    params = {"preset": "local_4gb", "task": proj.get("task_type", "classification"),
              "run_name": run_name, "project_id": pid, "model_id": proj.get("base_model"),
              "epochs": int(args.get("epochs", 2))}
    if proj.get("dataset_id"):
        meta = store.get_dataset_meta(pid, proj["dataset_id"]) or {}
        N = int(meta.get("n", 200))
        n_eval = max(4, min(100, int(N * 0.15)))
        params.update(dataset="workspace", dataset_id=proj["dataset_id"], n_train=N, n_eval=n_eval)
    else:
        params.update(dataset="bitext", label_field=args.get("label_field", "intent"),
                      n_train=700, n_eval=200)
    store.update_project(pid, plan_approved=True, status="training")
    try:
        jid = jm.start(params)
    except Exception as e:
        return {"error": str(e)}, None
    return ({"job_id": jid, "status": "started", "run_name": run_name,
             "note": "Training started; a few minutes. The user sees live progress."},
            {"type": "training_started", "job_id": jid, "run_name": run_name})


def _status(jm, args):
    job = jm.jobs.get(args.get("job_id") or jm.last_job_id)
    if not job:
        return {"error": "no training job has been started yet"}
    out = {"status": job.status}
    if job.result and job.result.get("task") == "classification":
        out["before_accuracy_pct"] = round(job.result["before"]["accuracy"] * 100, 1)
        out["after_accuracy_pct"] = round(job.result["after"]["accuracy"] * 100, 1)
        samples = job.result.get("samples", [])
        wins = [s for s in samples if s.get("tuned_ok") and not s.get("base_ok")]
        out["examples"] = [{"customer_message": s["input"][:160], "correct_label": s.get("gold"),
                            "guessed_before": s["base"], "guessed_after": s["tuned"]}
                           for s in (wins or samples)[:3]]
    prog = [e for e in job.events if e.get("type") == "progress"]
    if prog:
        out["latest_step"] = f"{prog[-1].get('step')}/{prog[-1].get('max_steps')}"
    return out


def _try_model(state, args):
    from server import model_service, store
    proj = _proj(state)
    run = (proj.get("active_run") if proj else None) or "agent"
    if not model_service.adapter_exists(run):
        return {"error": "No finished model yet — train one first."}
    return model_service.chat_with_run(run, args.get("message", ""), use_adapter=True)


def _auto_test(state, args):
    from server import model_service, store
    from finetune_studio import autotest
    proj = _proj(state)
    run = (proj.get("active_run") if proj else None) or "agent"
    if not model_service.adapter_exists(run):
        return {"error": "no trained model yet"}, None
    labels = []
    lp = pathlib.Path(Config().output_root) / run / "labels.json"
    if lp.exists():
        labels = json.loads(lp.read_text())
    task = proj.get("task_type", "classification") if proj else "classification"
    goal = (proj.get("goal") or proj.get("name", "")) if proj else ""
    card = autotest.run_autotest(run, task, goal, labels=labels, n=int(args.get("n", 10)))
    brief = {"score": card["score"], "correct": card["correct"], "total": card["total"], "verdict": card["verdict"]}
    return brief, {"type": "autotest", "card": card}


def _deploy_info(state, args):
    from finetune_studio import deploy
    proj = _proj(state)
    run = (proj.get("active_run") if proj else None) or "agent"
    info = deploy.export_info(run)
    if not info:
        return {"error": "no trained model to export yet"}, None
    brief = {"run_name": run, "size_mb": info["size_mb"], "base_model": info["base_model"]}
    return brief, {"type": "deploy", "info": info}


def _project_context(project_id) -> str | None:
    from server import store
    if not project_id:
        return None
    p = store.get_project(project_id)
    if not p:
        return None
    ds = ""
    if p.get("dataset_id"):
        meta = store.get_dataset_meta(p["id"], p["dataset_id"]) or {}
        ds = f" Dataset ready: {meta.get('n', 0)} examples."
    return (f"CURRENT PROJECT: {p['name']} (task={p['task_type']}, model={p['base_model']}, "
            f"status={p['status']}, plan_approved={p['plan_approved']}, active_run={p.get('active_run')}).{ds}")


def run_agent_turn(messages: list, jm, project_id: str | None = None, max_rounds: int = 6) -> dict:
    from server import settings
    model = settings.current_gemini_model()   # read live so admin changes apply immediately
    client = _client()
    convo = [{"role": "system", "content": SYSTEM}]
    ctx = _project_context(project_id)
    if ctx:
        convo.append({"role": "system", "content": ctx})
    convo += [{"role": m["role"], "content": m["content"]} for m in messages
              if m.get("role") in ("user", "assistant")]
    actions: list = []
    state = {"project_id": project_id}

    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model=model, messages=convo, tools=TOOLS, temperature=0.3, max_tokens=1500)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return {"reply": msg.content or "", "actions": actions, "project_id": state["project_id"]}

        convo.append({"role": "assistant", "content": msg.content or "",
                      "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            action = None
            if name == "create_project":
                result, action = _create_project(state, args)
            elif name == "propose_plan":
                result, action = _propose_plan(state, args)
            elif name == "generate_training_data":
                result, action = _generate_data(state, args)
            elif name == "start_training":
                result, action = _start_training(jm, state, args)
            elif name == "get_training_status":
                result = _status(jm, args)
            elif name == "try_model":
                result = _try_model(state, args)
            elif name == "auto_test":
                result, action = _auto_test(state, args)
            elif name == "get_deploy_info":
                result, action = _deploy_info(state, args)
            else:
                result = {"error": f"unknown tool {name}"}
            if action:
                actions.append(action)
            convo.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

    resp = client.chat.completions.create(model=model, messages=convo, temperature=0.3, max_tokens=1500)
    return {"reply": resp.choices[0].message.content or "", "actions": actions, "project_id": state["project_id"]}
