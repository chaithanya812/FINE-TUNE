"""Headless end-to-end run: load data -> train -> evaluate base vs tuned -> report.

This is the "does the whole engine work?" script. Once the venv install finishes:

    .venv\\Scripts\\python.exe scripts\\run_local.py                 # classification, intent
    .venv\\Scripts\\python.exe scripts\\run_local.py --label-field category
    .venv\\Scripts\\python.exe scripts\\run_local.py --dataset csv    # offline sample data
    .venv\\Scripts\\python.exe scripts\\run_local.py --task generation

It prints a ready-to-paste resume line at the end.
"""
import sys
import pathlib
import json
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import Config
from finetune_studio.data import load_data
from finetune_studio.trainer import train
from finetune_studio.evaluate import evaluate_base_vs_tuned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="local_4gb", choices=["local_4gb", "colab_t4"])
    ap.add_argument("--task", default="classification", choices=["classification", "generation"])
    ap.add_argument("--label-field", default="intent", choices=["intent", "category"])
    ap.add_argument("--dataset", default="bitext", choices=["bitext", "csv"])
    ap.add_argument("--model")
    ap.add_argument("--mode", choices=["qlora", "lora"])
    ap.add_argument("--n-train", type=int)
    ap.add_argument("--n-eval", type=int)
    ap.add_argument("--epochs", type=float)
    ap.add_argument("--run-name", default="run1")
    args = ap.parse_args()

    over = {"task": args.task, "label_field": args.label_field,
            "dataset": args.dataset, "run_name": args.run_name}
    if args.model:    over["model_id"] = args.model
    if args.mode:     over["mode"] = args.mode
    if args.n_train:  over["n_train"] = args.n_train
    if args.n_eval:   over["n_eval"] = args.n_eval
    if args.epochs:   over["epochs"] = args.epochs

    cfg = Config.preset(args.preset, **over)
    cfg.save()
    print(f"== Config ==\n  model={cfg.model_id}  mode={cfg.mode}  task={cfg.task}\n")

    print("== Loading data ==")
    bundle = load_data(cfg)
    print(f"  train={len(bundle.train_df)}  eval={len(bundle.eval_df)}  labels={len(bundle.labels)}\n")

    print("== Training ==")
    adapter = train(cfg, bundle, progress_cb=lambda d: print(f"  step {d['step']}/{d['max_steps']}  loss={d['loss']}"))

    print("\n== Evaluating base vs tuned ==")
    res = evaluate_base_vs_tuned(cfg, bundle, adapter, progress_cb=lambda d: print(f"  {d}"))
    (cfg.output_dir / "report.json").write_text(json.dumps(res, indent=2))
    print("\n" + json.dumps(res, indent=2))

    if cfg.task == "classification":
        b, a = res["before"]["accuracy"], res["after"]["accuracy"]
        print("\n" + "=" * 70)
        print(f"RESUME LINE:\n  Fine-tuned {cfg.model_id.split('/')[-1]} ({cfg.mode.upper()}) for customer-support")
        print(f"  intent classification, improving held-out accuracy from {b*100:.0f}% to {a*100:.0f}%")
        print(f"  ({res['after']['n']}-example test set) on a 4GB laptop GPU.")
        print("=" * 70)


if __name__ == "__main__":
    main()
