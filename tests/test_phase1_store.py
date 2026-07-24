"""Phase 1 acceptance tests for the content-addressed store (server/db.py).

Deterministic, self-contained: every test redirects the store to a fresh temp dir,
so the real store/ is never touched and the suite is safe to re-run. Runnable with
pytest OR directly:  .\\.venv\\Scripts\\python.exe tests\\test_phase1_store.py

Covers the Phase 1 acceptance criteria that don't need a GPU:
  * runs are pinned to dataset_version_id + config_hash
  * a crash between write and rename leaves NO corrupt blob (tmp+rename)
  * orphan .tmp files are never read
  * concurrent writers don't corrupt the DB (WAL + busy_timeout transactions)
"""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from server import db  # noqa: E402


def _fresh_store() -> pathlib.Path:
    """Point db at a brand-new temp store and (re)initialize it."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="ft_test_"))
    db.STORE = tmp
    db.DB_PATH = tmp / "meta.db"
    db.BLOBS = tmp / "blobs"
    db.DATASET_BLOBS = db.BLOBS / "datasets"
    db.EVALSET_BLOBS = db.BLOBS / "evalsets"
    db.RUN_BLOBS = db.BLOBS / "runs"
    db.EXPORTS = tmp / "exports"
    db._BLOB_ROOTS = {"datasets": db.DATASET_BLOBS, "evalsets": db.EVALSET_BLOBS}
    db._initialized = False
    db.init()
    return tmp


def _integrity_ok() -> bool:
    c = db._connect()
    try:
        return c.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        c.close()


def test_run_pinning():
    """A recorded run stores the exact dataset_version_id + config_hash (the jobs.py path)."""
    _fresh_store()
    db.insert_project({"id": "p_r", "name": "r"})
    dv = db.add_dataset_version("p_r", [{"input": "x", "target": "A"}], source="synthetic")
    cfg = {"model_id": "Qwen", "epochs": 3, "lr": 2e-4,
           "run_name": "p_r_v1", "project_id": "p_r", "dataset_id": dv["id"]}
    r = db.add_run("p_r", "p_r_v1", dataset_version_id=dv["id"], config=cfg,
                   metrics={"before": 0.3, "after": 0.9})
    assert r["dataset_version_id"] == dv["id"]
    assert r["config_hash"] == db.config_hash(cfg) and r["config_hash"]
    got = db.get_run("p_r_v1")
    assert got["dataset_version_id"] == dv["id"] and got["config_hash"] == r["config_hash"]
    # config_hash ignores environmental keys (same hyperparams -> same hash)
    assert db.config_hash(cfg) == db.config_hash({**cfg, "run_name": "other", "output_root": "/z"})


def test_crash_between_write_and_rename_leaves_no_corrupt_blob():
    """Simulate a kill at the rename point: no final blob, no orphan tmp, reads still clean."""
    _fresh_store()
    real_replace = os.replace

    def boom(*_a, **_k):
        raise OSError("simulated kill during rename")

    os.replace = boom
    try:
        raised = False
        try:
            db.write_blob("datasets", [{"input": "a", "target": "b"}])
        except OSError:
            raised = True
        assert raised, "the interrupted write must surface, not silently 'succeed'"
    finally:
        os.replace = real_replace

    assert list(db.DATASET_BLOBS.glob("*.jsonl")) == [], "no corrupt final blob may exist"
    assert list(db.DATASET_BLOBS.glob(".*tmp*")) == [], "tmp must be cleaned up"

    # the store is fully usable afterwards
    sha, n = db.write_blob("datasets", [{"input": "a", "target": "b"}])
    assert db.read_blob("datasets", sha) == [{"input": "a", "target": "b"}]


def test_orphan_tmp_is_never_read():
    """A leftover .tmp (from some crash) is invisible to content-addressed reads."""
    _fresh_store()
    sha, _ = db.write_blob("datasets", [{"input": "real", "target": "row"}])
    orphan = db.DATASET_BLOBS / f".{sha}.jsonl.tmp.deadbeef"
    orphan.write_text("{ this is half-written garbage", encoding="utf-8")
    # real content still reads fine; the only *.jsonl file is the committed blob
    assert db.read_blob("datasets", sha) == [{"input": "real", "target": "row"}]
    assert [p.name for p in db.DATASET_BLOBS.glob("*.jsonl")] == [f"{sha}.jsonl"]


def test_concurrent_dataset_writes_dont_corrupt():
    """30 threads writing dataset versions to one project: all land, DB stays intact."""
    _fresh_store()
    db.insert_project({"id": "p_c", "name": "c"})
    errors: list[Exception] = []

    def worker(i: int):
        try:
            db.add_dataset_version("p_c", [{"input": f"i{i}", "target": "t"}], source="t")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert len(db.list_dataset_versions("p_c")) == 30
    assert _integrity_ok()


def test_concurrent_updates_to_one_row_dont_corrupt():
    """Many threads patching the SAME project row: serialized cleanly, no corruption."""
    _fresh_store()
    db.insert_project({"id": "p_u", "name": "u"})
    errors: list[Exception] = []

    def worker(i: int):
        try:
            db.update_project_fields("p_u", {"status": f"s{i}"})
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert db.fetch_project("p_u")["status"].startswith("s")
    assert _integrity_ok()


ALL = [test_run_pinning,
       test_crash_between_write_and_rename_leaves_no_corrupt_blob,
       test_orphan_tmp_is_never_read,
       test_concurrent_dataset_writes_dont_corrupt,
       test_concurrent_updates_to_one_row_dont_corrupt]


if __name__ == "__main__":
    failed = 0
    for t in ALL:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {e!r}")
    print(f"\n{len(ALL) - failed}/{len(ALL)} passed")
    sys.exit(1 if failed else 0)
