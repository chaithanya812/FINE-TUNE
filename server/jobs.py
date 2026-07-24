"""Background job manager - runs one training+eval pipeline at a time.

On a 4GB GPU we can only fit one model, so only one job runs at once. The training
thread appends typed events to job.events; the WebSocket in app.py tails that list
and streams them to the browser (simple, GIL-safe, no async plumbing across threads).
"""
from __future__ import annotations
import sys
import pathlib
import threading
import traceback
import uuid
import json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import Config
from finetune_studio.data import load_data
from finetune_studio.trainer import train, TrainingCancelled
from finetune_studio.evaluate import evaluate_base_vs_tuned

_OVERRIDE_KEYS = ["task", "label_field", "dataset", "model_id", "mode",
                  "n_train", "n_eval", "epochs", "run_name", "lora_r", "lr",
                  "project_id", "dataset_id", "max_seq_len"]


def _overrides(p: dict) -> dict:
    o = {k: p[k] for k in _OVERRIDE_KEYS if p.get(k) not in (None, "")}
    if p.get("model"):          # accept "model" as an alias for "model_id"
        o["model_id"] = p["model"]
    return o


class Job:
    def __init__(self, jid: str, params: dict):
        self.id = jid
        self.params = params
        self.status = "queued"          # queued | running | done | error | cancelled
        self.events: list[dict] = []
        self.result = None
        self.run_name = params.get("run_name", "ui_run")
        self.cancel = threading.Event()

    def emit(self, ev: dict):
        self.events.append(ev)


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.active: str | None = None
        self.last_job_id: str | None = None   # so the agent can answer "how did it do?"

    def is_busy(self) -> bool:
        return bool(self.active and self.jobs.get(self.active) and self.jobs[self.active].status == "running")

    def start(self, params: dict) -> str:
        with self.lock:
            if self.is_busy():
                raise RuntimeError("A training job is already running (4GB GPU = one at a time).")
            jid = uuid.uuid4().hex[:8]
            job = Job(jid, params)
            self.jobs[jid] = job
            self.active = jid
            self.last_job_id = jid
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return jid

    def cancel(self, job_id: str) -> bool:
        """Signal a queued/running job to stop. Returns False if it can't be cancelled."""
        job = self.jobs.get(job_id)
        if job and job.status in ("queued", "running"):
            job.cancel.set()
            return True
        return False

    def _run(self, job: Job):
        job.status = "running"
        try:
            from server import model_service
            model_service.unload()   # free any warm playground cache before training (4GB)
            cfg = Config.preset(job.params.get("preset", "local_4gb"), **_overrides(job.params))
            cfg.save()
            job.emit({"type": "log", "msg": f"Config · {cfg.model_id.split('/')[-1]} · {cfg.mode.upper()} · {cfg.task}"})

            job.emit({"type": "log", "msg": "Loading data…"})
            bundle = load_data(cfg)
            (cfg.output_dir / "labels.json").write_text(json.dumps(bundle.labels))
            job.emit({"type": "log", "msg": f"Loaded {len(bundle.train_df)} train / {len(bundle.eval_df)} eval · {len(bundle.labels)} labels"})

            # Show the user a few real examples the model is about to learn from.
            job.emit({"type": "train_samples", "samples": [
                {"input": str(r["input"])[:140], "label": str(r["target"])[:140]}
                for _, r in bundle.train_df.head(3).iterrows()
            ]})

            job.emit({"type": "log", "msg": "Training… (forward → loss → backward → update)"})
            adapter = train(
                cfg, bundle,
                progress_cb=lambda d: job.emit({"type": "progress", **d}),
                cancel_event=job.cancel,
                resume=bool(job.params.get("resume", False)),
            )

            job.emit({"type": "log", "msg": "Training complete. Scoring base vs tuned on held-out data…"})

            def ecb(d):
                ph = d.get("phase")
                if ph:
                    job.emit({"type": "log", "msg": "Scoring " + ("tuned model…" if ph == "eval_after" else "base model…")})

            res = evaluate_base_vs_tuned(cfg, bundle, adapter, progress_cb=ecb)
            job.result = res
            (cfg.output_dir / "report.json").write_text(json.dumps(res, indent=2))

            # Record this run against its project (history / versioning).
            pid = job.params.get("project_id")
            if pid:
                from dataclasses import asdict
                from server import store, db
                summary = {"run_name": cfg.run_name, "task": res.get("task")}
                if res.get("task") == "classification":
                    summary.update(before=round(res["before"]["accuracy"], 4),
                                   after=round(res["after"]["accuracy"], 4),
                                   delta=round(res.get("delta_accuracy", 0.0), 4),
                                   before_ci=[res["before"].get("ci_lo"), res["before"].get("ci_hi")],
                                   after_ci=[res["after"].get("ci_lo"), res["after"].get("ci_hi")],
                                   n=res["after"].get("n"),
                                   mcnemar_p=(res.get("mcnemar_base_vs_tuned") or {}).get("p_value"))
                # Pin the run to the exact data + config + frozen eval set that
                # produced it, and snapshot its config/report immutably.
                store.add_run(pid, summary, dataset_version_id=cfg.dataset_id,
                              config=asdict(cfg), eval_set_id=res.get("eval_set_id"))
                db.save_run_snapshot(cfg.run_name, asdict(cfg), res)
                store.update_project(pid, status="evaluated")

            job.emit({"type": "result", "report": res, "run_name": cfg.run_name})
            job.emit({"type": "done", "run_name": cfg.run_name})
            job.status = "done"
        except TrainingCancelled:
            job.emit({"type": "cancelled", "msg": "Training stopped by you. The GPU has been freed."})
            job.status = "cancelled"
        except Exception as e:
            job.emit({"type": "error", "msg": str(e)})
            job.status = "error"
            traceback.print_exc()
        finally:
            try:
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
