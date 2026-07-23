"""File-based workspace store — projects, datasets, and run records on disk.

Deliberately simple (JSON files, no DB) for a single-user local app. Layout:

    workspaces/<project_id>/project.json          # the project manifest
    workspaces/<project_id>/datasets/<id>.jsonl   # dataset rows ({input, target})
    workspaces/<project_id>/datasets/<id>.json    # dataset metadata (stats, split)

Trained adapters still live in runs/<run_name>/ as before; a project references its
run_names, so we get history/versioning without moving the engine's output.
"""
from __future__ import annotations
import json
import shutil
import threading
import time
import uuid
from pathlib import Path

from config import PROJECT_ROOT

WORKSPACES = PROJECT_ROOT / "workspaces"
_LOCK = threading.RLock()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _pdir(pid: str) -> Path:
    return WORKSPACES / pid


def _pfile(pid: str) -> Path:
    return _pdir(pid) / "project.json"


# --------------------------------------------------------------------- projects
def create_project(name: str, goal: str = "", task_type: str = "classification",
                   base_model: str = "Qwen/Qwen2.5-0.5B-Instruct", method: str = "qlora") -> dict:
    pid = "p_" + uuid.uuid4().hex[:8]
    proj = {
        "id": pid,
        "name": name or "Untitled project",
        "goal": goal,
        "task_type": task_type,
        "base_model": base_model,
        "method": method,
        "config": {},           # training overrides (label_field, epochs, n_train, ...)
        "plan": None,           # the approved Plan Card
        "plan_approved": False,
        "dataset_id": None,     # active dataset
        "status": "draft",      # draft|planned|data_ready|training|evaluated|deployed
        "runs": [],             # [{run_name, version, task, before, after, delta, created_at}]
        "active_run": None,
        "iterations": [],       # improvement-loop history
        "created_at": _now(),
        "updated_at": _now(),
    }
    with _LOCK:
        (_pdir(pid) / "datasets").mkdir(parents=True, exist_ok=True)
        _pfile(pid).write_text(json.dumps(proj, indent=2))
    return proj


def get_project(pid: str) -> dict | None:
    f = _pfile(pid)
    return json.loads(f.read_text()) if f.exists() else None


def list_projects() -> list[dict]:
    if not WORKSPACES.exists():
        return []
    out = []
    for d in sorted(WORKSPACES.iterdir(), reverse=True):
        f = d / "project.json"
        if f.exists():
            p = json.loads(f.read_text())
            out.append({k: p.get(k) for k in
                        ("id", "name", "task_type", "base_model", "status",
                         "created_at", "updated_at", "active_run")})
    return out


_PATCHABLE = {"name", "goal", "task_type", "base_model", "method", "config",
              "plan", "plan_approved", "dataset_id", "status", "active_run", "iterations"}


def update_project(pid: str, **fields) -> dict | None:
    with _LOCK:
        proj = get_project(pid)
        if not proj:
            return None
        for k, v in fields.items():
            if k not in _PATCHABLE:
                continue
            if k == "config" and isinstance(v, dict):
                proj.setdefault("config", {}).update(v)   # deep-merge config
            else:
                proj[k] = v
        proj["updated_at"] = _now()
        _pfile(pid).write_text(json.dumps(proj, indent=2))
    return proj


def delete_project(pid: str) -> bool:
    d = _pdir(pid)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        return True
    return False


def add_run(pid: str, run_record: dict) -> dict | None:
    """Append a run record (auto-stamped with version + created_at) and mark it active."""
    with _LOCK:
        proj = get_project(pid)
        if not proj:
            return None
        rec = {"created_at": _now(), **run_record, "version": len(proj.get("runs", [])) + 1}
        proj.setdefault("runs", []).append(rec)
        proj["active_run"] = rec.get("run_name")
        proj["updated_at"] = _now()
        _pfile(pid).write_text(json.dumps(proj, indent=2))
    return proj


# --------------------------------------------------------------------- datasets
def save_dataset(pid: str, rows: list[dict], meta: dict, dataset_id: str | None = None) -> str:
    did = dataset_id or ("d_" + uuid.uuid4().hex[:8])
    ddir = _pdir(pid) / "datasets"
    ddir.mkdir(parents=True, exist_ok=True)
    with (ddir / f"{did}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {**meta, "id": did, "n": len(rows), "created_at": _now()}
    (ddir / f"{did}.json").write_text(json.dumps(meta, indent=2))
    return did


def get_dataset_rows(pid: str, did: str) -> list[dict]:
    f = _pdir(pid) / "datasets" / f"{did}.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def get_dataset_meta(pid: str, did: str) -> dict | None:
    f = _pdir(pid) / "datasets" / f"{did}.json"
    return json.loads(f.read_text()) if f.exists() else None


def list_datasets(pid: str) -> list[dict]:
    ddir = _pdir(pid) / "datasets"
    if not ddir.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(ddir.glob("*.json"))]
