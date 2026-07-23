"""Task-type registry — the thing that de-hardcodes us from "customer support only".

Every dataset row is normalized to {input, target}. A TaskType knows how to turn
that pair into chat messages for training, how to prompt at inference, and which
eval metric proves it worked. Add a task type here and the whole pipeline (train /
evaluate / chat / synth) picks it up.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass
class TaskType:
    key: str
    label: str
    description: str
    metric: str          # "accuracy" (classification) | "judge" (open-ended) | "json_match"
    system: str          # {labels} is filled in for classification
    example_target: str  # a hint shown in the UI for what "target" means


TASKS: dict[str, TaskType] = {
    "classification": TaskType(
        "classification", "Classification",
        "Sort each message into exactly one of a fixed set of labels.",
        "accuracy",
        "You are a classifier. Read the input and reply with exactly one label from this "
        "list, and nothing else:\n{labels}",
        "the correct label",
    ),
    "generation": TaskType(
        "generation", "Reply / Generation",
        "Write a good free-form reply to each message.",
        "judge",
        "You are a helpful, friendly assistant. Answer the user's message clearly and "
        "appropriately in a few sentences.",
        "the ideal reply",
    ),
    "tone": TaskType(
        "tone", "Tone / Style rewrite",
        "Rewrite the input text in a target voice or style, keeping the meaning.",
        "judge",
        "You rewrite text in the requested style while preserving its meaning. "
        "Output only the rewritten text, nothing else.",
        "the rewritten text",
    ),
    "extraction": TaskType(
        "extraction", "Data extraction",
        "Pull structured fields out of messy input as JSON.",
        "json_match",
        "You extract structured data. Read the input and output ONLY a compact JSON object "
        "with the requested fields — no prose, no code fences.",
        "the JSON to extract",
    ),
}


def get(key: str) -> TaskType:
    return TASKS.get(key) or TASKS["classification"]


def _system(task_key: str, labels=None) -> str:
    t = get(task_key)
    if task_key == "classification":
        return t.system.format(labels=", ".join(labels or []))
    return t.system


def build_train_messages(task_key: str, row: dict, labels=None) -> list[dict]:
    """Full (system, user, assistant) chat used for TRAINING."""
    return [
        {"role": "system", "content": _system(task_key, labels)},
        {"role": "user", "content": str(row["input"])},
        {"role": "assistant", "content": str(row["target"])},
    ]


def build_prompt_messages(task_key: str, input_text, labels=None) -> list[dict]:
    """Prompt only (system, user) used for INFERENCE / EVAL."""
    return [
        {"role": "system", "content": _system(task_key, labels)},
        {"role": "user", "content": str(input_text)},
    ]


def as_dicts() -> list[dict]:
    return [asdict(t) for t in TASKS.values()]
