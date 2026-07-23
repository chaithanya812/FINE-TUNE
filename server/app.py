"""FastAPI backend for Fine-Tune Studio.

    .venv\\Scripts\\python.exe -m uvicorn server.app:app --port 8000
    # then open http://localhost:8000

Endpoints:
  GET  /                -> the web UI
  POST /api/train       -> start a training job, returns {job_id}
  WS   /ws/{job_id}     -> live stream of training/eval events
  POST /api/chat        -> chat with a finished model (adapter on/off)
  GET  /api/runs        -> list previous runs and their reports
"""
from __future__ import annotations
import sys
import pathlib
import json
import asyncio

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import Config
from server.jobs import JobManager
from server import model_service
from server import system as system_info
from server import store
from server import settings as app_settings
from server.agent import run_agent_turn, agent_available

ROOT = pathlib.Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

app = FastAPI(title="Fine-Tune Studio")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"], allow_headers=["*"],
)
jm = JobManager()


# The static UI is mounted at the site root at the END of this file (after the
# API routes, so it doesn't shadow them). Assets are served from web/.


# --------------------------------------------------------------------- train
class TrainReq(BaseModel):
    preset: str = "local_4gb"
    task: str = "classification"
    label_field: str = "intent"
    dataset: str = "bitext"
    model_id: str | None = None
    mode: str | None = None
    epochs: float | None = None
    n_train: int | None = None
    n_eval: int | None = None
    run_name: str = "ui_run"
    resume: bool = False
    project_id: str | None = None
    dataset_id: str | None = None


@app.get("/api/system")
def system_ep():
    """GPU probe + model recommendation for the hardware badge and VRAM bar."""
    return system_info.system_summary()


@app.get("/api/tasks")
def tasks_ep():
    """The task types the studio supports (classification, generation, tone, extraction)."""
    from finetune_studio import tasks as task_registry
    return {"tasks": task_registry.as_dicts()}


# ------------------------------------------------------------------- settings
class SettingsReq(BaseModel):
    gemini_model: str | None = None


@app.get("/api/settings")
def get_settings_ep():
    """Current app settings (which Gemini teacher model is active)."""
    return app_settings.get_settings()


@app.put("/api/settings")
@app.post("/api/settings")
def update_settings_ep(req: SettingsReq):
    """Change the active Gemini teacher model. Applies immediately (no restart)."""
    return app_settings.update_settings(gemini_model=req.gemini_model)


@app.get("/api/gemini/models")
def gemini_models_ep():
    """Live list of text models the user's key can call, for the admin picker."""
    return app_settings.list_gemini_models()


@app.get("/api/templates")
def templates_ep():
    """Use-case templates for the 'what do you want to build?' gallery."""
    from finetune_studio import templates as tmpl
    return {"templates": tmpl.as_dicts()}


# ------------------------------------------------------------------ projects
class ProjectReq(BaseModel):
    name: str
    goal: str = ""
    task_type: str = "classification"
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    method: str = "qlora"


@app.post("/api/projects")
def create_project_ep(req: ProjectReq):
    return store.create_project(**req.model_dump())


@app.get("/api/projects")
def list_projects_ep():
    return {"projects": store.list_projects()}


@app.get("/api/projects/{pid}")
def get_project_ep(pid: str):
    p = store.get_project(pid)
    return p or JSONResponse({"error": "not found"}, status_code=404)


@app.patch("/api/projects/{pid}")
def update_project_ep(pid: str, patch: dict):
    p = store.update_project(pid, **patch)
    return p or JSONResponse({"error": "not found"}, status_code=404)


@app.delete("/api/projects/{pid}")
def delete_project_ep(pid: str):
    return {"deleted": store.delete_project(pid)}


# --------------------------------------------------------------- data services
class UploadReq(BaseModel):
    text: str
    filename: str = ""
    input_col: str | None = None
    target_col: str | None = None


class GenerateReq(BaseModel):
    goal: str
    n: int = 50
    labels: list[str] | None = None
    seeds: list[dict] | None = None
    task: str | None = None


class EstimateReq(BaseModel):
    n: int = 50
    task: str = "classification"
    labels: list[str] | None = None


def _finalize_dataset(pid: str, rows: list, source: str, task: str) -> dict:
    from finetune_studio import dataio
    rows = dataio.dedup(rows)
    st = dataio.stats(rows, task)
    st["split"] = dataio.split_counts(rows)
    did = store.save_dataset(pid, rows, {"source": source, "task": task, **st})
    store.update_project(pid, dataset_id=did, status="data_ready")
    return {"dataset_id": did, "stats": st, "preview": rows[:12], "n": len(rows)}


@app.post("/api/synth/estimate")
def synth_estimate_ep(req: EstimateReq):
    from finetune_studio import synth
    return synth.estimate(req.n, req.task, req.labels)


@app.post("/api/projects/{pid}/data/upload")
def data_upload_ep(pid: str, req: UploadReq):
    proj = store.get_project(pid)
    if not proj:
        return JSONResponse({"error": "project not found"}, status_code=404)
    from finetune_studio import dataio
    parsed = dataio.parse_upload(req.text, req.filename, req.input_col, req.target_col)
    if not parsed["rows"]:
        return JSONResponse({"error": "no rows parsed", "detail": parsed["errors"],
                             "columns": parsed.get("columns")}, status_code=400)
    result = _finalize_dataset(pid, parsed["rows"], "upload", proj.get("task_type", "classification"))
    result["errors"] = parsed["errors"]
    result["columns"] = parsed.get("columns")
    return result


@app.post("/api/projects/{pid}/data/generate")
def data_generate_ep(pid: str, req: GenerateReq):
    proj = store.get_project(pid)
    if not proj:
        return JSONResponse({"error": "project not found"}, status_code=404)
    from finetune_studio import synth
    if not synth.available():
        return JSONResponse({"error": "No GEMINI_API_KEY set — add it to .env to generate data."}, status_code=400)
    task = req.task or proj.get("task_type", "classification")
    gen = synth.generate(task, req.goal, n=req.n, labels=req.labels, seeds=req.seeds)
    if not gen["rows"]:
        return JSONResponse({"error": "generation produced no rows", "detail": gen["errors"]}, status_code=502)
    result = _finalize_dataset(pid, gen["rows"], "synthetic", task)
    result["errors"] = gen["errors"]
    return result


@app.get("/api/projects/{pid}/data/{did}")
def data_get_ep(pid: str, did: str, full: bool = False):
    meta = store.get_dataset_meta(pid, did)
    if not meta:
        return JSONResponse({"error": "dataset not found"}, status_code=404)
    rows = store.get_dataset_rows(pid, did)
    return {"meta": meta, "preview": rows[:20],
            "rows": rows if full else None, "n": len(rows)}


class DatasetEditReq(BaseModel):
    rows: list[dict]


@app.put("/api/projects/{pid}/data/{did}")
def data_save_ep(pid: str, did: str, req: DatasetEditReq):
    """Save the user's edited dataset back (overwrites the same dataset id) and
    recompute stats. This is what powers the spreadsheet-style editor."""
    proj = store.get_project(pid)
    if not proj:
        return JSONResponse({"error": "project not found"}, status_code=404)
    task = proj.get("task_type", "classification")
    rows = [{"input": str(r.get("input", "")).strip(), "target": str(r.get("target", "")).strip()}
            for r in req.rows if str(r.get("input", "")).strip()]
    if not rows:
        return JSONResponse({"error": "no rows to save (every row needs an input)"}, status_code=400)
    from finetune_studio import dataio
    st = dataio.stats(rows, task)
    st["split"] = dataio.split_counts(rows)
    store.save_dataset(pid, rows, {"source": "edited", "task": task, **st}, dataset_id=did)
    store.update_project(pid, dataset_id=did, status="data_ready")
    return {"dataset_id": did, "stats": st, "n": len(rows)}


@app.get("/api/projects/{pid}/colab")
def colab_ep(pid: str):
    """Download a one-click Unsloth Colab notebook with this project's data baked in."""
    proj = store.get_project(pid)
    if not proj:
        return JSONResponse({"error": "project not found"}, status_code=404)
    did = proj.get("dataset_id")
    rows = store.get_dataset_rows(pid, did) if did else []
    from finetune_studio import colab_export
    from config import PROJECT_ROOT
    out = PROJECT_ROOT / "workspaces" / pid / f"{pid}_colab.ipynb"
    colab_export.write_project_notebook(proj, rows, path=out)
    return FileResponse(str(out), filename=f"{proj.get('name', 'model')}_colab.ipynb",
                        media_type="application/x-ipynb+json")


# -------------------------------------------------------------- auto-test (QA)
class AutotestReq(BaseModel):
    run_name: str | None = None
    n: int = 10


@app.post("/api/projects/{pid}/autotest")
def autotest_ep(pid: str, req: AutotestReq):
    proj = store.get_project(pid)
    if not proj:
        return JSONResponse({"error": "project not found"}, status_code=404)
    if jm.is_busy():
        return JSONResponse({"error": "training in progress — try again once it finishes"}, status_code=409)
    from finetune_studio import autotest
    run_name = req.run_name or proj.get("active_run")
    if not run_name or not model_service.adapter_exists(run_name):
        return JSONResponse({"error": "no trained model yet for this project"}, status_code=404)
    labels = []
    lp = pathlib.Path(Config().output_root) / run_name / "labels.json"
    if lp.exists():
        labels = json.loads(lp.read_text())
    return autotest.run_autotest(run_name, proj.get("task_type", "classification"),
                                 proj.get("goal") or proj.get("name", ""), labels=labels, n=req.n)


# -------------------------------------------------------------- deploy / export
@app.get("/api/runs/{run_name}/export")
def export_ep(run_name: str):
    from finetune_studio import deploy
    info = deploy.export_info(run_name)
    return info or JSONResponse({"error": "no adapter for that run"}, status_code=404)


@app.get("/api/runs/{run_name}/download")
def download_ep(run_name: str):
    from finetune_studio import deploy
    z = deploy.zip_adapter(run_name)
    if not z:
        return JSONResponse({"error": "no adapter"}, status_code=404)
    return FileResponse(str(z), filename=f"{run_name}_adapter.zip", media_type="application/zip")


class CompareReq(BaseModel):
    a: str
    b: str


@app.post("/api/runs/compare")
def compare_ep(req: CompareReq):
    from finetune_studio import deploy
    return deploy.compare(req.a, req.b)


@app.delete("/api/runs/{run_name}")
def delete_run_ep(run_name: str):
    from finetune_studio import deploy
    return {"deleted": deploy.delete_run(run_name)}


@app.post("/api/runs/{run_name}/judge")
def judge_run_ep(run_name: str):
    """Score a finished generation run's STORED held-out samples with the LLM judge
    (no GPU / model load — pure API calls). Fills before/after judge scores into
    report.json so old unscored runs get real numbers retroactively."""
    rep_path = pathlib.Path(Config().output_root) / run_name / "report.json"
    if not rep_path.exists():
        return JSONResponse({"error": "no report for that run"}, status_code=404)
    report = json.loads(rep_path.read_text(encoding="utf-8"))
    if report.get("task") == "classification":
        return JSONResponse({"error": "classification runs already have accuracy scores"}, status_code=400)
    samples = report.get("samples") or []
    if not samples:
        return JSONResponse({"error": "no stored samples to score"}, status_code=400)
    from finetune_studio import judge
    cfg = Config()
    if not judge.available(cfg):
        return JSONResponse({"error": "no judge available — set GEMINI_API_KEY (or JUDGE_API_KEY) in .env"},
                            status_code=400)
    b_scores, a_scores, scored = [], [], []
    for s in samples:
        b = judge.score_reply(cfg, s.get("input", ""), s.get("base", ""))
        a = judge.score_reply(cfg, s.get("input", ""), s.get("tuned", ""))
        if b is not None:
            b_scores.append(b)
        if a is not None:
            a_scores.append(a)
        scored.append({**s, "base_score": b, "tuned_score": a})

    def _avg(xs):
        return round(sum(xs) / len(xs), 2) if xs else None

    report.setdefault("before", {})["avg_judge_score"] = _avg(b_scores)
    report.setdefault("after", {})["avg_judge_score"] = _avg(a_scores)
    report["samples"] = scored
    report["judged_n"] = len(scored)
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"run_name": run_name, "before_avg": _avg(b_scores), "after_avg": _avg(a_scores),
            "n": len(scored), "samples": scored}


@app.post("/api/train")
def start_train(req: TrainReq):
    try:
        return {"job_id": jm.start(req.model_dump())}
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=409)


@app.post("/api/train/{job_id}/cancel")
def cancel_train(job_id: str):
    if jm.cancel(job_id):
        return {"status": "cancelling"}
    return JSONResponse({"error": "job is not running"}, status_code=404)


@app.websocket("/ws/{job_id}")
async def ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    job = jm.jobs.get(job_id)
    if not job:
        await websocket.send_json({"type": "error", "msg": "unknown job"})
        await websocket.close()
        return
    cursor = 0
    while True:
        while cursor < len(job.events):
            await websocket.send_json(job.events[cursor])
            cursor += 1
        if job.status in ("done", "error", "cancelled"):
            break
        await asyncio.sleep(0.4)
    await websocket.close()


@app.get("/api/status/{job_id}")
def job_status(job_id: str):
    job = jm.jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return {"status": job.status, "events": job.events, "result": job.result}


# ---------------------------------------------------------------------- runs
@app.get("/api/runs")
def list_runs():
    runs = []
    root = Config().output_root
    p = pathlib.Path(root)
    if p.exists():
        for d in sorted(p.iterdir()):
            rep = d / "report.json"
            if rep.exists():
                runs.append({"run_name": d.name, "report": json.loads(rep.read_text())})
    return {"runs": runs}


@app.get("/api/resumable")
def resumable_ep():
    """Runs with saved checkpoints but no finished report — interrupted by a
    crash/shutdown. The UI offers 'Resume from step N?' on next open."""
    out = []
    root = pathlib.Path(Config().output_root)
    if root.exists():
        for d in sorted(root.iterdir()):
            ck = d / "checkpoints"
            ckpts = sorted(ck.glob("checkpoint-*")) if ck.exists() else []
            if ckpts and not (d / "report.json").exists():
                last = max((int(c.name.split("-")[-1]) for c in ckpts if c.name.split("-")[-1].isdigit()), default=0)
                cfgp = d / "config.json"
                out.append({"run_name": d.name, "last_step": last,
                            "config": json.loads(cfgp.read_text()) if cfgp.exists() else {}})
    return {"resumable": out}


# ---------------------------------------------------------------------- chat
class ChatReq(BaseModel):
    run_name: str = "ui_run"
    message: str
    use_adapter: bool = True


@app.post("/api/chat")
def chat_ep(req: ChatReq):
    if jm.is_busy():
        return JSONResponse({"error": "Training in progress — the GPU is busy. Try again once it finishes."}, status_code=409)
    if not model_service.adapter_exists(req.run_name):
        return JSONResponse({"error": f"No trained model found for run '{req.run_name}'. Train one first."}, status_code=404)
    return model_service.chat_with_run(req.run_name, req.message, req.use_adapter)


# ------------------------------------------------------------------ AI agent
class AgentReq(BaseModel):
    messages: list  # conversation so far: [{role: "user"|"assistant", content: str}, ...]
    project_id: str | None = None


@app.post("/api/agent")
def agent_ep(req: AgentReq):
    if not agent_available():
        return JSONResponse({"error": "Set GEMINI_API_KEY in .env to enable the agent."}, status_code=400)
    try:
        return run_agent_turn(req.messages, jm, project_id=req.project_id)
    except Exception as e:
        return JSONResponse({"error": f"agent error: {e}"}, status_code=500)


# Serve the legacy vanilla UI (web/) at the site root. The primary UI is the
# Next.js + shadcn app in webui/ (run separately on :3000).
app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")
