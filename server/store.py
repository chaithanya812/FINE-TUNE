"""Compatibility layer: the legacy file-based store API, now backed by db.py.

The studio used to keep projects and datasets as loose JSON/JSONL files under
workspaces/<pid>/. That layer allowed in-place mutation (editing a dataset
overwrote its file), so runs could not be pinned to the exact data that produced
them. Phase 1 of the evals overhaul replaced storage with an immutable,
content-addressed store indexed by SQLite (server/db.py).

This module keeps the OLD 10-function API and its legacy dict shapes so every
existing endpoint (app.py, agent.py, jobs.py, data.py) keeps working unchanged.
New code should prefer db.py directly. The important behaviour change is that
save_dataset() now mints a NEW immutable dataset version (with a parent link)
instead of overwriting — nothing is ever mutated in place again.

Mapping from legacy concepts to the new store:
    legacy project["dataset_id"]  <-> projects.dataset_version_id (the active version)
    legacy project["runs"] list   <-> the runs table (reconstructed per project)
    a dataset "did"               <-> a dataset_versions.id (immutable content blob)
"""
from __future__ import annotations

import uuid

from server import db


# --------------------------------------------------------- legacy-shape helpers
def _legacy_project(pid: str) -> dict | None:
    """Reassemble the full legacy project dict the old file store returned."""
    p = db.fetch_project(pid)
    if p is None:
        return None
    runs = [
        {**r["metrics"], "run_name": r["id"], "version": r["version"],
         "created_at": r["created_at"]}
        for r in db.list_runs(pid)
    ]
    return {
        "id": p["id"], "name": p["name"], "goal": p["goal"],
        "task_type": p["task_type"], "base_model": p["base_model"], "method": p["method"],
        "config": p["config"], "plan": p["plan"], "plan_approved": p["plan_approved"],
        "dataset_id": p["dataset_version_id"],       # legacy name for the active version
        "status": p["status"], "runs": runs, "active_run": p["active_run"],
        "iterations": p["iterations"],
        "created_at": p["created_at"], "updated_at": p["updated_at"],
    }


def _legacy_meta(dv: dict | None) -> dict | None:
    """Reassemble the legacy dataset meta dict (domain meta + id/n/created_at)."""
    if dv is None:
        return None
    meta = dict(dv.get("meta") or {})
    meta.setdefault("source", dv.get("source", ""))
    meta["id"] = dv["id"]
    meta["n"] = dv["n_rows"]
    meta["created_at"] = dv["created_at"]
    return meta


# --------------------------------------------------------------------- projects
def create_project(name: str, goal: str = "", task_type: str = "classification",
                   base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
                   method: str = "qlora") -> dict:
    pid = "p_" + uuid.uuid4().hex[:8]
    db.insert_project({
        "id": pid, "name": name or "Untitled project", "goal": goal,
        "task_type": task_type, "base_model": base_model, "method": method,
        "status": "draft", "plan_approved": False, "config": {},
        "plan": None, "iterations": [],
    })
    return _legacy_project(pid)


def get_project(pid: str) -> dict | None:
    return _legacy_project(pid)


def list_projects() -> list[dict]:
    return [
        {k: p.get(k) for k in ("id", "name", "task_type", "base_model", "status",
                               "created_at", "updated_at", "active_run")}
        for p in db.fetch_projects()
    ]


# legacy patchable set (unknown keys ignored, exactly like the old store)
_PATCHABLE = {"name", "goal", "task_type", "base_model", "method", "config",
              "plan", "plan_approved", "dataset_id", "status", "active_run", "iterations"}


def update_project(pid: str, **fields) -> dict | None:
    patch = {k: v for k, v in fields.items() if k in _PATCHABLE}
    if "dataset_id" in patch:                       # legacy name -> new column
        patch["dataset_version_id"] = patch.pop("dataset_id")
    if db.update_project_fields(pid, patch) is None:
        return None
    return _legacy_project(pid)


def delete_project(pid: str) -> bool:
    # Index rows are removed; content-addressed blobs and any migrated
    # workspaces/<pid> backup are left untouched.
    return db.delete_project(pid)


def add_run(pid: str, run_record: dict, dataset_version_id: str | None = None,
            config: dict | None = None, eval_set_id: str | None = None) -> dict | None:
    """Append a run PINNED to the data + config + eval set that produced it, mark active.

    `dataset_version_id`, `config`, and `eval_set_id` are the provenance pins
    (jobs.py passes them); without them the run is still recorded, just without
    full provenance (legacy calls).
    """
    if db.fetch_project(pid) is None:
        return None
    run_name = run_record.get("run_name") or ("run_" + uuid.uuid4().hex[:8])
    db.add_run(pid, run_name, dataset_version_id=dataset_version_id, config=config,
               eval_set_id=eval_set_id, metrics=run_record)
    db.update_project_fields(pid, {"active_run": run_name})
    return _legacy_project(pid)


# --------------------------------------------------------------------- datasets
def save_dataset(pid: str, rows: list[dict], meta: dict, dataset_id: str | None = None) -> str:
    """Save a dataset as a NEW immutable version and return its id.

    Unlike the old store, this never overwrites: passing `dataset_id` means "this
    was edited FROM that version", so the new version records it as parent_id
    (lineage). Callers should use the RETURNED id as the project's active dataset.
    """
    dv = db.add_dataset_version(
        pid, rows, meta,
        parent_id=dataset_id,
        source=str(meta.get("source", "")),
    )
    return dv["id"]


def get_dataset_rows(pid: str, did: str) -> list[dict]:
    return db.get_dataset_version_rows(did)


def get_dataset_meta(pid: str, did: str) -> dict | None:
    return _legacy_meta(db.get_dataset_version(did))


def list_datasets(pid: str) -> list[dict]:
    return [_legacy_meta(dv) for dv in db.list_dataset_versions(pid)]
