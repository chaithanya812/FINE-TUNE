"""SQLite index + content-addressed blob store for Fine-Tune Studio.

Provenance foundation (Phase 1 of the evals/storage overhaul). Everything the
studio produces becomes immutable and pinned:

    store/
      meta.db                        # the index (tables below); stdlib sqlite3 only
      blobs/datasets/<sha256>.jsonl  # IMMUTABLE dataset versions (content-addressed)
      blobs/evalsets/<sha256>.jsonl  # IMMUTABLE frozen exams (Phase 2 fills these)
      blobs/runs/<run_id>/           # per-run immutable metadata snapshot (config, report)
      exports/                       # disposable generated artifacts (colab, etc.)

Design rules (do NOT "simplify" away):
  * Blobs are content-addressed: the file name IS the sha256 of its bytes, so a
    given content can never be silently mutated. Editing a dataset writes a NEW
    blob + a NEW dataset_versions row whose parent_id points at the source
    version (a lineage DAG). Old blobs are never touched.
  * Every blob write is atomic: tmp file -> flush + fsync -> os.replace. A crash
    mid-write leaves an orphan .tmp, never a half-written blob (readers only ever
    open the final content-addressed path).
  * All DB access goes through short-lived connections in explicit transactions,
    with WAL + busy_timeout so concurrent writers wait instead of corrupting.
  * No ORM, no migration framework -- stdlib sqlite3 and a schema_version int.

store.py is a thin compatibility layer over this module so the existing HTTP
endpoints keep working; new code can call db.py directly.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from config import PROJECT_ROOT

# --------------------------------------------------------------------- layout
STORE = PROJECT_ROOT / "store"
DB_PATH = STORE / "meta.db"
BLOBS = STORE / "blobs"
DATASET_BLOBS = BLOBS / "datasets"
EVALSET_BLOBS = BLOBS / "evalsets"
RUN_BLOBS = BLOBS / "runs"
EXPORTS = STORE / "exports"

SCHEMA_VERSION = 3

# config keys that don't change the trained artifact -> excluded from config_hash,
# so the same hyper-parameters hash identically across machines and run names.
_VOLATILE_CONFIG_KEYS = {"run_name", "output_root", "csv_path", "project_id", "dataset_id"}

_BLOB_ROOTS = {"datasets": DATASET_BLOBS, "evalsets": EVALSET_BLOBS}

_INIT_LOCK = threading.Lock()
_initialized = False

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS projects (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    goal                TEXT DEFAULT '',
    task_type           TEXT DEFAULT 'classification',
    base_model          TEXT DEFAULT '',
    method              TEXT DEFAULT 'qlora',
    status              TEXT DEFAULT 'draft',
    dataset_version_id  TEXT,               -- active dataset version -> dataset_versions.id
    active_run          TEXT,
    plan_approved       INTEGER DEFAULT 0,
    config_json         TEXT DEFAULT '{}',  -- training overrides
    plan_json           TEXT,               -- approved Plan Card (nullable)
    iterations_json     TEXT DEFAULT '[]',  -- improvement-loop history
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_versions (
    id          TEXT PRIMARY KEY,     -- stable handle (old d_xxxx ids preserved on migrate)
    project_id  TEXT NOT NULL,
    sha256      TEXT NOT NULL,        -- content id -> blobs/datasets/<sha256>.jsonl
    parent_id   TEXT,                 -- lineage: the version this was edited from
    n_rows      INTEGER NOT NULL,
    source      TEXT DEFAULT '',      -- synthetic|edited|imported|migrated
    meta_json   TEXT DEFAULT '{}',    -- classes, split, balance_warning, ...
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dv_project ON dataset_versions(project_id);

CREATE TABLE IF NOT EXISTS eval_sets (
    id                           TEXT PRIMARY KEY,
    project_id                   TEXT NOT NULL,
    sha256                       TEXT NOT NULL,   -- -> blobs/evalsets/<sha256>.jsonl
    n_rows                       INTEGER NOT NULL,
    created_from_dataset_version TEXT,
    meta_json                    TEXT DEFAULT '{}',
    created_at                   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_es_project ON eval_sets(project_id);

CREATE TABLE IF NOT EXISTS runs (
    id                  TEXT PRIMARY KEY,   -- run_name (e.g. p_af913690_v1)
    project_id          TEXT NOT NULL,
    dataset_version_id  TEXT,               -- the exact data that produced this run
    eval_set_id         TEXT,               -- frozen exam it was scored on (null until Phase 2)
    config_hash         TEXT,               -- semantic hash of the training config
    status              TEXT DEFAULT 'done',
    version             INTEGER,            -- per-project ordinal
    metrics_json        TEXT DEFAULT '{}',  -- before/after/delta summary
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_id);

CREATE TABLE IF NOT EXISTS tests (
    id           TEXT PRIMARY KEY,
    project_id   TEXT NOT NULL,
    input        TEXT NOT NULL,
    expected     TEXT,
    origin       TEXT DEFAULT 'manual',  -- perturbation|generated|manual|past_failure
    perturbation TEXT,                    -- which transform, when origin=perturbation
    severity     INTEGER DEFAULT 1,       -- 1 = worst (high-confidence wrong)
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tests_project ON tests(project_id);

CREATE TABLE IF NOT EXISTS judgments (
    hash        TEXT PRIMARY KEY,       -- sha256(judge_model, input, replies, orientation)
    verdict     TEXT NOT NULL,          -- A|B|tie (or a rubric json)
    judge_model TEXT,
    created_at  TEXT NOT NULL
);
"""


# --------------------------------------------------------------------- helpers
def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _connect() -> sqlite3.Connection:
    """A fresh connection tuned for safe concurrent single-process access."""
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")     # readers don't block the writer
    conn.execute("PRAGMA synchronous=NORMAL")   # durable enough for WAL, much faster
    conn.execute("PRAGMA busy_timeout=5000")    # wait on a lock instead of erroring
    return conn


def init() -> None:
    """Create store dirs + schema (idempotent). Call before any DB use."""
    global _initialized
    if _initialized:
        return
    with _INIT_LOCK:
        if _initialized:
            return
        for d in (STORE, DATASET_BLOBS, EVALSET_BLOBS, RUN_BLOBS, EXPORTS):
            d.mkdir(parents=True, exist_ok=True)
        conn = _connect()
        try:
            with conn:
                conn.executescript(_SCHEMA)
                row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
                if row is None:
                    conn.execute("INSERT INTO schema_version(version) VALUES (?)",
                                 (SCHEMA_VERSION,))
                elif row[0] < SCHEMA_VERSION:   # additive tables only -> just bump the marker
                    conn.execute("UPDATE schema_version SET version=?", (SCHEMA_VERSION,))
        finally:
            conn.close()
        _initialized = True


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    """Write bytes atomically: tmp in the same dir -> fsync -> os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{uuid.uuid4().hex}"
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)          # atomic on Windows (ReplaceFile) + POSIX
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _rows_to_jsonl(rows: list[dict]) -> bytes:
    """Deterministic serialization: sort keys so identical content hashes identically."""
    body = "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows)
    return (body + ("\n" if rows else "")).encode("utf-8")


def _jsonl_to_rows(data: bytes) -> list[dict]:
    return [json.loads(l) for l in data.decode("utf-8").splitlines() if l.strip()]


def config_hash(config: dict | None) -> str:
    """Stable 16-hex hash of the *semantic* training config (environmental keys stripped)."""
    clean = {k: v for k, v in (config or {}).items() if k not in _VOLATILE_CONFIG_KEYS}
    return _sha256(json.dumps(clean, sort_keys=True, ensure_ascii=False).encode("utf-8"))[:16]


# ----------------------------------------------------------------------- blobs
def write_blob(kind: str, rows: list[dict]) -> tuple[str, int]:
    """Content-address rows into blobs/<kind>/<sha256>.jsonl. Returns (sha256, n_rows).

    Dedup: identical content already on disk is left as-is (never rewritten).
    """
    init()
    root = _BLOB_ROOTS[kind]
    data = _rows_to_jsonl(rows)
    sha = _sha256(data)
    path = root / f"{sha}.jsonl"
    if not path.exists():
        _atomic_write(path, data)
    return sha, len(rows)


def read_blob(kind: str, sha256: str) -> list[dict]:
    root = _BLOB_ROOTS[kind]
    path = root / f"{sha256}.jsonl"
    return _jsonl_to_rows(path.read_bytes()) if path.exists() else []


# -------------------------------------------------------------------- projects
def _project_from_row(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"], "name": r["name"], "goal": r["goal"],
        "task_type": r["task_type"], "base_model": r["base_model"], "method": r["method"],
        "status": r["status"], "dataset_version_id": r["dataset_version_id"],
        "active_run": r["active_run"], "plan_approved": bool(r["plan_approved"]),
        "config": json.loads(r["config_json"] or "{}"),
        "plan": json.loads(r["plan_json"]) if r["plan_json"] else None,
        "iterations": json.loads(r["iterations_json"] or "[]"),
        "created_at": r["created_at"], "updated_at": r["updated_at"],
    }


def insert_project(proj: dict) -> dict:
    """Insert a project row from a normalized dict (idempotent via INSERT OR REPLACE)."""
    init()
    now = _now()
    conn = _connect()
    try:
        with conn:
            conn.execute(
                """INSERT OR REPLACE INTO projects
                   (id, name, goal, task_type, base_model, method, status,
                    dataset_version_id, active_run, plan_approved, config_json,
                    plan_json, iterations_json, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (proj["id"], proj.get("name") or "Untitled project", proj.get("goal", ""),
                 proj.get("task_type", "classification"), proj.get("base_model", ""),
                 proj.get("method", "qlora"), proj.get("status", "draft"),
                 proj.get("dataset_version_id"), proj.get("active_run"),
                 int(bool(proj.get("plan_approved", False))),
                 json.dumps(proj.get("config", {})),
                 json.dumps(proj["plan"]) if proj.get("plan") is not None else None,
                 json.dumps(proj.get("iterations", [])),
                 proj.get("created_at") or now, proj.get("updated_at") or now),
            )
    finally:
        conn.close()
    return fetch_project(proj["id"])


def fetch_project(pid: str) -> dict | None:
    init()
    conn = _connect()
    try:
        r = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
        return _project_from_row(r) if r else None
    finally:
        conn.close()


def fetch_projects() -> list[dict]:
    init()
    conn = _connect()
    try:
        rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
        return [_project_from_row(r) for r in rows]
    finally:
        conn.close()


_PROJECT_COLUMNS = {
    "name": "name", "goal": "goal", "task_type": "task_type", "base_model": "base_model",
    "method": "method", "status": "status", "dataset_version_id": "dataset_version_id",
    "active_run": "active_run",
}
_PROJECT_JSON_COLUMNS = {"plan": "plan_json", "iterations": "iterations_json"}


def update_project_fields(pid: str, fields: dict) -> dict | None:
    """Patch selected project columns. `config` deep-merges; others replace."""
    init()
    conn = _connect()
    try:
        with conn:
            cur = conn.execute("SELECT config_json FROM projects WHERE id=?", (pid,)).fetchone()
            if cur is None:
                return None
            sets: list[str] = []
            vals: list = []
            for k, v in fields.items():
                if k in _PROJECT_COLUMNS:
                    sets.append(f"{_PROJECT_COLUMNS[k]}=?"); vals.append(v)
                elif k == "plan_approved":
                    sets.append("plan_approved=?"); vals.append(int(bool(v)))
                elif k == "config" and isinstance(v, dict):
                    merged = json.loads(cur["config_json"] or "{}"); merged.update(v)
                    sets.append("config_json=?"); vals.append(json.dumps(merged))
                elif k in _PROJECT_JSON_COLUMNS:
                    sets.append(f"{_PROJECT_JSON_COLUMNS[k]}=?")
                    vals.append(json.dumps(v) if v is not None else None)
            if sets:
                sets.append("updated_at=?"); vals.append(_now())
                vals.append(pid)
                conn.execute(f"UPDATE projects SET {', '.join(sets)} WHERE id=?", vals)
    finally:
        conn.close()
    return fetch_project(pid)


def delete_project(pid: str) -> bool:
    """Delete a project and its index rows. Content-addressed blobs are left in
    place (immutable + possibly shared via dedup); a GC pass can reclaim them later."""
    init()
    conn = _connect()
    try:
        with conn:
            conn.execute("DELETE FROM runs WHERE project_id=?", (pid,))
            conn.execute("DELETE FROM dataset_versions WHERE project_id=?", (pid,))
            conn.execute("DELETE FROM eval_sets WHERE project_id=?", (pid,))
            cur = conn.execute("DELETE FROM projects WHERE id=?", (pid,))
        return cur.rowcount > 0
    finally:
        conn.close()


# ------------------------------------------------------------- dataset versions
def _dv_from_row(r: sqlite3.Row) -> dict:
    return {"id": r["id"], "project_id": r["project_id"], "sha256": r["sha256"],
            "parent_id": r["parent_id"], "n_rows": r["n_rows"], "source": r["source"],
            "meta": json.loads(r["meta_json"] or "{}"), "created_at": r["created_at"]}


def add_dataset_version(project_id: str, rows: list[dict], meta: dict | None = None,
                        parent_id: str | None = None, source: str = "",
                        dataset_id: str | None = None,
                        created_at: str | None = None) -> dict:
    """Write an IMMUTABLE dataset version: content-address the rows into a blob and
    record a new row. `parent_id` links the version this was edited from (lineage).
    `created_at` lets a migration preserve the original timestamp."""
    init()
    sha, n = write_blob("datasets", rows)
    did = dataset_id or ("d_" + uuid.uuid4().hex[:8])
    created = created_at or _now()
    meta = dict(meta or {})
    conn = _connect()
    try:
        with conn:
            conn.execute(
                """INSERT INTO dataset_versions
                   (id, project_id, sha256, parent_id, n_rows, source, meta_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (did, project_id, sha, parent_id, n, source, json.dumps(meta), created),
            )
    finally:
        conn.close()
    return {"id": did, "project_id": project_id, "sha256": sha, "parent_id": parent_id,
            "n_rows": n, "source": source, "meta": meta, "created_at": created}


def get_dataset_version(did: str) -> dict | None:
    init()
    conn = _connect()
    try:
        r = conn.execute("SELECT * FROM dataset_versions WHERE id=?", (did,)).fetchone()
        return _dv_from_row(r) if r else None
    finally:
        conn.close()


def get_dataset_version_rows(did: str) -> list[dict]:
    dv = get_dataset_version(did)
    return read_blob("datasets", dv["sha256"]) if dv else []


def list_dataset_versions(project_id: str) -> list[dict]:
    init()
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM dataset_versions WHERE project_id=? ORDER BY created_at",
            (project_id,)).fetchall()
        return [_dv_from_row(r) for r in rows]
    finally:
        conn.close()


# ------------------------------------------------------------------- eval sets
def _es_from_row(r: sqlite3.Row) -> dict:
    return {"id": r["id"], "project_id": r["project_id"], "sha256": r["sha256"],
            "n_rows": r["n_rows"],
            "created_from_dataset_version": r["created_from_dataset_version"],
            "meta": json.loads(r["meta_json"] or "{}"), "created_at": r["created_at"]}


def add_eval_set(project_id: str, rows: list[dict], meta: dict | None = None,
                 created_from_dataset_version: str | None = None,
                 eval_set_id: str | None = None) -> dict:
    """Write an IMMUTABLE frozen eval set (Phase 2 defines how they're built)."""
    init()
    sha, n = write_blob("evalsets", rows)
    esid = eval_set_id or ("es_" + uuid.uuid4().hex[:8])
    created = _now()
    meta = dict(meta or {})
    conn = _connect()
    try:
        with conn:
            conn.execute(
                """INSERT INTO eval_sets
                   (id, project_id, sha256, n_rows, created_from_dataset_version,
                    meta_json, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (esid, project_id, sha, n, created_from_dataset_version,
                 json.dumps(meta), created),
            )
    finally:
        conn.close()
    return {"id": esid, "project_id": project_id, "sha256": sha, "n_rows": n,
            "created_from_dataset_version": created_from_dataset_version,
            "meta": meta, "created_at": created}


def get_eval_set(esid: str) -> dict | None:
    init()
    conn = _connect()
    try:
        r = conn.execute("SELECT * FROM eval_sets WHERE id=?", (esid,)).fetchone()
        return _es_from_row(r) if r else None
    finally:
        conn.close()


def get_eval_set_rows(esid: str) -> list[dict]:
    es = get_eval_set(esid)
    return read_blob("evalsets", es["sha256"]) if es else []


def list_eval_sets(project_id: str) -> list[dict]:
    init()
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM eval_sets WHERE project_id=? ORDER BY created_at",
            (project_id,)).fetchall()
        return [_es_from_row(r) for r in rows]
    finally:
        conn.close()


# ------------------------------------------------------------------------ runs
def _run_from_row(r: sqlite3.Row) -> dict:
    return {"id": r["id"], "project_id": r["project_id"],
            "dataset_version_id": r["dataset_version_id"], "eval_set_id": r["eval_set_id"],
            "config_hash": r["config_hash"], "status": r["status"], "version": r["version"],
            "metrics": json.loads(r["metrics_json"] or "{}"), "created_at": r["created_at"]}


def add_run(project_id: str, run_id: str, *, dataset_version_id: str | None = None,
            eval_set_id: str | None = None, config: dict | None = None,
            config_hash_value: str | None = None, status: str = "done",
            metrics: dict | None = None, version: int | None = None,
            created_at: str | None = None) -> dict:
    """Record a run PINNED to the exact data + config that produced it.

    Pass `config` to derive config_hash automatically, or `config_hash_value`
    directly. `created_at` lets a migration preserve the original run time.
    Idempotent on run_id (INSERT OR REPLACE) so re-migration is safe.
    """
    init()
    chash = config_hash_value if config_hash_value is not None else (
        config_hash(config) if config is not None else None)
    created = created_at or _now()
    conn = _connect()
    try:
        with conn:
            if version is None:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM runs WHERE project_id=?", (project_id,)).fetchone()
                version = int(row["c"]) + 1
            conn.execute(
                """INSERT OR REPLACE INTO runs
                   (id, project_id, dataset_version_id, eval_set_id, config_hash,
                    status, version, metrics_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (run_id, project_id, dataset_version_id, eval_set_id, chash,
                 status, version, json.dumps(metrics or {}), created),
            )
    finally:
        conn.close()
    return {"id": run_id, "project_id": project_id, "dataset_version_id": dataset_version_id,
            "eval_set_id": eval_set_id, "config_hash": chash, "status": status,
            "version": version, "metrics": metrics or {}, "created_at": created}


def get_run(run_id: str) -> dict | None:
    init()
    conn = _connect()
    try:
        r = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return _run_from_row(r) if r else None
    finally:
        conn.close()


def list_runs(project_id: str) -> list[dict]:
    init()
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM runs WHERE project_id=? ORDER BY version", (project_id,)).fetchall()
        return [_run_from_row(r) for r in rows]
    finally:
        conn.close()


_RUN_COLUMNS = {"dataset_version_id": "dataset_version_id", "eval_set_id": "eval_set_id",
                "config_hash": "config_hash", "status": "status", "version": "version"}


def update_run(run_id: str, **fields) -> dict | None:
    """Patch run columns (e.g. stamp eval_set_id in Phase 2, update metrics later)."""
    init()
    conn = _connect()
    try:
        with conn:
            if conn.execute("SELECT 1 FROM runs WHERE id=?", (run_id,)).fetchone() is None:
                return None
            sets: list[str] = []
            vals: list = []
            for k, v in fields.items():
                if k in _RUN_COLUMNS:
                    sets.append(f"{_RUN_COLUMNS[k]}=?"); vals.append(v)
                elif k == "metrics":
                    sets.append("metrics_json=?"); vals.append(json.dumps(v or {}))
            if sets:
                vals.append(run_id)
                conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id=?", vals)
    finally:
        conn.close()
    return get_run(run_id)


# --------------------------------------------------------------- run artifacts
def run_blob_dir(run_id: str) -> Path:
    d = RUN_BLOBS / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run_snapshot(run_id: str, config: dict, report: dict) -> None:
    """Immutable metadata snapshot of a run (small + greppable). The trained
    adapter continues to live in runs/<run_id>/ -- we only snapshot the pins."""
    d = run_blob_dir(run_id)
    _atomic_write(d / "config.json",
                  json.dumps(config, indent=2, ensure_ascii=False).encode("utf-8"))
    _atomic_write(d / "report.json",
                  json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8"))


# ----------------------------------------------------------- regression bank
def _test_from_row(r: sqlite3.Row) -> dict:
    return {"id": r["id"], "project_id": r["project_id"], "input": r["input"],
            "expected": r["expected"], "origin": r["origin"],
            "perturbation": r["perturbation"], "severity": r["severity"],
            "created_at": r["created_at"]}


def add_test(project_id: str, input: str, expected: str | None = None,
             origin: str = "manual", perturbation: str | None = None,
             severity: int = 1, test_id: str | None = None) -> dict:
    """Add a case to the permanent regression bank (idempotent on test_id)."""
    init()
    tid = test_id or ("t_" + uuid.uuid4().hex[:8])
    created = _now()
    conn = _connect()
    try:
        with conn:
            conn.execute(
                """INSERT OR REPLACE INTO tests
                   (id, project_id, input, expected, origin, perturbation, severity, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (tid, project_id, input, expected, origin, perturbation, int(severity), created))
    finally:
        conn.close()
    return {"id": tid, "project_id": project_id, "input": input, "expected": expected,
            "origin": origin, "perturbation": perturbation, "severity": int(severity),
            "created_at": created}


def list_tests(project_id: str) -> list[dict]:
    """The regression bank for a project, worst (severity 1) first."""
    init()
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM tests WHERE project_id=? ORDER BY severity, created_at",
            (project_id,)).fetchall()
        return [_test_from_row(r) for r in rows]
    finally:
        conn.close()


def bank_failure(project_id: str, input: str, expected: str | None = None,
                 origin: str = "past_failure", perturbation: str | None = None,
                 severity: int = 1) -> dict | None:
    """Bank a failing case IF it isn't already there (the bank only grows, never
    duplicates). Dedup by (project, input, expected). Returns the new row, or None."""
    init()
    conn = _connect()
    try:
        dup = conn.execute(
            "SELECT id FROM tests WHERE project_id=? AND input=? AND IFNULL(expected,'')=IFNULL(?,'')",
            (project_id, input, expected)).fetchone()
    finally:
        conn.close()
    if dup:
        return None
    return add_test(project_id, input, expected, origin=origin,
                    perturbation=perturbation, severity=severity)


# ------------------------------------------------------- judgment cache (judge v2)
def cache_get(hash_key: str) -> str | None:
    """Return a cached judge verdict, or None. Makes re-running an eval near-free."""
    init()
    conn = _connect()
    try:
        r = conn.execute("SELECT verdict FROM judgments WHERE hash=?", (hash_key,)).fetchone()
        return r["verdict"] if r else None
    finally:
        conn.close()


def cache_put(hash_key: str, verdict: str, judge_model: str | None = None) -> None:
    init()
    conn = _connect()
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO judgments (hash, verdict, judge_model, created_at) "
                "VALUES (?,?,?,?)", (hash_key, verdict, judge_model, _now()))
    finally:
        conn.close()
