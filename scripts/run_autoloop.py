"""Auto-improve: keep training (escalating the recipe) until a target is hit.

    .venv\\Scripts\\python.exe scripts\\run_autoloop.py --target 0.85 --max-attempts 3

Each attempt trains, evaluates base-vs-tuned, and stops early once the target
held-out accuracy is reached. Prints the full journey and the winning run.
"""
import sys
import pathlib
import json
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import Config
from finetune_studio.autoloop import auto_improve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="local_4gb", choices=["local_4gb", "colab_t4"])
    ap.add_argument("--task", default="classification", choices=["classification", "generation"])
    ap.add_argument("--label-field", default="intent", choices=["intent", "category"])
    ap.add_argument("--dataset", default="bitext", choices=["bitext", "csv"])
    ap.add_argument("--target", type=float, default=0.85)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--run-name", default="auto")
    args = ap.parse_args()

    cfg = Config.preset(args.preset, task=args.task, label_field=args.label_field,
                        dataset=args.dataset, run_name=args.run_name, n_train=600, epochs=2.0)

    def step_cb(d):
        if d.get("loss") is not None:
            print(f"    step {d['step']}/{d['max_steps']}  loss={d['loss']}")

    def event_cb(d):
        if d["type"] == "attempt":
            print(f"\n=== Attempt {d['attempt']}/{d['max']}  (recipe changes: {d['overrides'] or 'baseline'}) ===")
        else:
            mark = "TARGET HIT" if d["hit"] else "below target"
            print(f"--- Attempt {d['attempt']} result: {d['metric']*100:.1f}%  ({mark}, target {d['target']*100:.0f}%)")

    res = auto_improve(cfg, target=args.target, max_attempts=args.max_attempts,
                       step_cb=step_cb, event_cb=event_cb)

    print("\n" + "=" * 70)
    print("JOURNEY:")
    for h in res["history"]:
        print(f"  attempt {h['attempt']}: {h['metric']*100:5.1f}%   ({h['run_name']})")
    best = res["best"]
    print(f"\nBEST: attempt {best['attempt']} -> {best['metric']*100:.1f}%  "
          f"({'target reached' if res['hit_target'] else 'target not reached'})")
    print(f"Winning model: runs/{best['run_name']}/adapter")
    print("=" * 70)

    pathlib.Path("autoloop_report.json").write_text(json.dumps(
        {"history": res["history"], "best_run": best["run_name"],
         "best_metric": best["metric"], "hit_target": res["hit_target"]}, indent=2))


if __name__ == "__main__":
    main()
