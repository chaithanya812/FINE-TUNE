"""Generate a Colab notebook you can upload to Google Colab and Run All.

    .venv\\Scripts\\python.exe scripts\\make_colab.py
    .venv\\Scripts\\python.exe scripts\\make_colab.py --model Qwen/Qwen2.5-3B-Instruct --task generation

Writes finetune_colab.ipynb to the project root.
"""
import sys
import pathlib
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import Config
from finetune_studio.colab_export import write_notebook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--task", default="classification", choices=["classification", "generation"])
    ap.add_argument("--label-field", default="intent", choices=["intent", "category"])
    ap.add_argument("--epochs", type=float)
    args = ap.parse_args()

    over = {"task": args.task, "label_field": args.label_field}
    if args.model:
        over["model_id"] = args.model
    if args.epochs:
        over["epochs"] = args.epochs

    cfg = Config.preset("colab_t4", **over)
    path = write_notebook(cfg)
    print(f"Wrote {path}")
    print("Upload it to https://colab.research.google.com  ->  Runtime: T4 GPU  ->  Run all.")


if __name__ == "__main__":
    main()
