"""Migrate the legacy workspaces/ + runs/ file store into the content-addressed store.

Idempotent and non-destructive: reads workspaces/<pid>/ and runs/<name>/ ONLY and
leaves them untouched as a backup. Re-running skips anything already imported (use
--force to overwrite existing project rows from the on-disk manifest).

Every dataset version present on disk is imported (history preserved); each run
listed by a project is pinned to the exact dataset_version + config_hash recorded
in its runs/<name>/config.json, and its config/report are snapshotted immutably.

Run:
    .\\.venv\\Scripts\\python.exe scripts\\migrate_store.py [--force]
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import PROJECT_ROOT          # noqa: E402
from server import db                    # noqa: E402

WORKSPACES = PROJECT_ROOT / "workspaces"
RUNS = PROJECT_ROOT / "runs"


def _load_json(p: pathlib.Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  ! could not read {p}: {e}")
        return None


def _load_jsonl(p: pathlib.Path) -> list[dict]:
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def _read_run_file(run_name: str, fname: str) -> dict:
    """Load runs/<run_name>/<fname> as a dict, or {} if absent/unreadable."""
    p = RUNS / run_name / fname
    return (_load_json(p) or {}) if p.exists() else {}


def migrate(force: bool = False) -> dict:
    db.init()
    s = {"projects": 0, "projects_skipped": 0, "datasets": 0, "datasets_skipped": 0,
         "runs": 0, "snapshots": 0, "warnings": []}
    if not WORKSPACES.exists():
        print("no workspaces/ dir — nothing to migrate")
        return s

    for pdir in sorted(WORKSPACES.iterdir()):
        if not pdir.is_dir():
            continue
        pj = pdir / "project.json"
        if not pj.exists():
            continue
        proj = _load_json(pj)
        if not proj:
            continue
        pid = proj.get("id") or pdir.name

        # --- project row -----------------------------------------------------
        if db.fetch_project(pid) and not force:
            s["projects_skipped"] += 1
            print(f"= project {pid} already imported — skipping row (datasets/runs still synced)")
        else:
            db.insert_project({
                "id": pid,
                "name": proj.get("name", pid),
                "goal": proj.get("goal", ""),
                "task_type": proj.get("task_type", "classification"),
                "base_model": proj.get("base_model", ""),
                "method": proj.get("method", "qlora"),
                "status": proj.get("status", "draft"),
                "dataset_version_id": proj.get("dataset_id"),
                "active_run": proj.get("active_run"),
                "plan_approved": proj.get("plan_approved", False),
                "config": proj.get("config", {}),
                "plan": proj.get("plan"),
                "iterations": proj.get("iterations", []),
                "created_at": proj.get("created_at"),
                "updated_at": proj.get("updated_at"),
            })
            s["projects"] += 1
            print(f"+ project {pid}  ({proj.get('name', '')})")

        # --- datasets: import every version on disk, preserving its id -------
        ddir = pdir / "datasets"
        if ddir.exists():
            for meta_file in sorted(ddir.glob("*.json")):
                did = meta_file.stem
                if db.get_dataset_version(did):
                    s["datasets_skipped"] += 1
                    continue
                jsonl = ddir / f"{did}.jsonl"
                if not jsonl.exists():
                    s["warnings"].append(f"{pid}: dataset {did} has meta but no .jsonl rows")
                    continue
                rows = _load_jsonl(jsonl)
                meta = _load_json(meta_file) or {}
                db.add_dataset_version(pid, rows, meta, parent_id=None,
                                       source=str(meta.get("source", "migrated")),
                                       dataset_id=did, created_at=meta.get("created_at"))
                s["datasets"] += 1
                print(f"    · dataset {did}  ({len(rows)} rows)")

        # --- runs: pinned from runs/<name>/config.json ----------------------
        for rec in proj.get("runs", []):
            run_name = rec.get("run_name")
            if not run_name:
                continue
            rcfg = _read_run_file(run_name, "config.json")
            rreport = _read_run_file(run_name, "report.json")
            dvid = rcfg.get("dataset_id") or proj.get("dataset_id")
            chash = db.config_hash(rcfg) if rcfg else None
            metrics = {k: v for k, v in rec.items() if k not in ("version", "created_at")}
            db.add_run(pid, run_name, dataset_version_id=dvid, config_hash_value=chash,
                       status="done", metrics=metrics, version=rec.get("version"),
                       created_at=rec.get("created_at"))
            s["runs"] += 1
            if rcfg or rreport:
                db.save_run_snapshot(run_name, rcfg, rreport)
                s["snapshots"] += 1
            print(f"    * run {run_name}  (dv={dvid} cfg={chash})")

    return s


if __name__ == "__main__":
    force = "--force" in sys.argv
    print(f"Migrating legacy store -> content-addressed store (force={force})\n")
    s = migrate(force=force)
    print("\n--- summary ---")
    for k in ("projects", "projects_skipped", "datasets", "datasets_skipped", "runs", "snapshots"):
        print(f"  {k:18} {s[k]}")
    if s["warnings"]:
        print("  warnings:")
        for w in s["warnings"]:
            print("    -", w)
    print("\nOld workspaces/ and runs/ left untouched (backup). Done.")
