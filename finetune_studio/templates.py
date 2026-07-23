"""Use-case templates — the "what do you want to build?" gallery.

This is the fix for the single biggest confusion: people ask for "a chatbot" and get
a 3-way classifier. Each template makes the *shape* of the model explicit up front —
does it ROUTE messages, ANSWER them, EXTRACT from them, or REWRITE them? — and carries
sane defaults (task type, base model, how much data, which metric) plus an honest note
about what fine-tuning can and can't do for that use case.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass
class Template:
    key: str
    title: str
    tagline: str            # one line, plain English
    task_type: str          # maps to tasks.py
    what_it_does: str       # concrete behaviour
    honest_note: str        # what fine-tuning will / won't achieve here
    example_labels: list    # for classification templates (starter categories)
    example_rows: list      # 2-3 {input, target} seeds shown in the UI
    suggested_n: int        # how many examples to aim for a *good* result
    min_n: int              # below this it won't really work
    needs_bigger_model: bool  # true => steer to the Colab/cloud path
    local_model: str
    colab_model: str
    data_hint: str          # what to collect from the user


TEMPLATES: dict[str, Template] = {
    "router": Template(
        key="router", title="Router / Classifier",
        tagline="Sort every message into one of your buckets (intents, topics, moderation).",
        task_type="classification",
        what_it_does="Reads a message and replies with exactly one label. Great for routing "
                     "support tickets, tagging leads, or flagging content.",
        honest_note="This does NOT answer questions — it only picks a label. A tiny model is "
                    "perfect here and will happily beat a big LLM on speed and cost.",
        example_labels=["Lead Generation", "General Enquiry", "Feedback", "Complaint"],
        example_rows=[
            {"input": "Can someone call me about bulk pricing for 50 seats?", "target": "Lead Generation"},
            {"input": "What are your holiday hours?", "target": "General Enquiry"},
            {"input": "The new dashboard is so much faster, love it!", "target": "Feedback"},
        ],
        suggested_n=300, min_n=90, needs_bigger_model=False,
        local_model="Qwen/Qwen2.5-0.5B-Instruct", colab_model="Qwen/Qwen2.5-3B-Instruct",
        data_hint="Name your categories (3-8 works best) and, if you have them, a few real messages per category.",
    ),
    "assistant": Template(
        key="assistant", title="Assistant / Q&A in your voice",
        tagline="Answer questions and chat in your product's style and tone.",
        task_type="generation",
        what_it_does="Writes free-form replies the way YOU would — your tone, your format, "
                     "your do's and don'ts.",
        honest_note="Fine-tuning teaches STYLE and FORMAT, not facts. For it to know your "
                    "specific docs/prices, you'll want RAG (retrieval) on top — that's on the roadmap. "
                    "A good assistant also needs a bigger model, so this steers you to the Colab/cloud path.",
        example_labels=[],
        example_rows=[
            {"input": "Do you offer refunds?", "target": "Absolutely — you can request a full refund within 30 days, no questions asked. Want me to start one for you?"},
            {"input": "how do i reset my password", "target": "No worries! Head to Settings → Security → Reset password, and you'll get an email link within a minute."},
        ],
        suggested_n=400, min_n=150, needs_bigger_model=True,
        local_model="Qwen/Qwen2.5-0.5B-Instruct", colab_model="Qwen/Qwen2.5-3B-Instruct",
        data_hint="Collect real question→ideal-answer pairs in your brand voice. The more real examples, the better.",
    ),
    "extractor": Template(
        key="extractor", title="Data extractor",
        tagline="Turn messy text into clean, structured JSON.",
        task_type="extraction",
        what_it_does="Reads unstructured input (an email, a note, an order) and outputs a tidy "
                     "JSON object with just the fields you care about.",
        honest_note="Very reliable use case for small models — the output is checkable field-by-field, "
                    "so you get honest accuracy numbers, not vibes.",
        example_labels=[],
        example_rows=[
            {"input": "Hey, ship 3 blue mugs to Priya at 12 MG Road, Bangalore 560001, rush please",
             "target": '{"item":"blue mug","qty":3,"name":"Priya","city":"Bangalore","pincode":"560001","priority":"rush"}'},
            {"input": "invoice #A-882 total 4,500 due 30 Jul", "target": '{"invoice":"A-882","total":4500,"due":"2026-07-30"}'},
        ],
        suggested_n=300, min_n=100, needs_bigger_model=False,
        local_model="Qwen/Qwen2.5-0.5B-Instruct", colab_model="Qwen/Qwen2.5-3B-Instruct",
        data_hint="Decide the exact JSON fields you want, then give messy input → correct JSON pairs.",
    ),
    "rewriter": Template(
        key="rewriter", title="Tone / Style rewriter",
        tagline="Rewrite text into a target voice while keeping the meaning.",
        task_type="tone",
        what_it_does="Takes any input and rewrites it — blunt → polite, formal → casual, "
                     "long → short — in a consistent house style.",
        honest_note="A focused, achievable task for a small model. Give clear before→after pairs "
                    "and it locks onto your style fast.",
        example_labels=[],
        example_rows=[
            {"input": "Send the report now.", "target": "Hi! Whenever you get a chance, could you send over the report? Thanks so much."},
            {"input": "This is wrong, fix it.", "target": "I think there might be a small mistake here — could we take another look together?"},
        ],
        suggested_n=250, min_n=80, needs_bigger_model=False,
        local_model="Qwen/Qwen2.5-0.5B-Instruct", colab_model="Qwen/Qwen2.5-3B-Instruct",
        data_hint="Collect original → rewritten pairs that show the exact tone you want.",
    ),
    "reasoner": Template(
        key="reasoner", title="Domain reasoner (advanced)",
        tagline="Numbers, tables, multi-step reasoning over your domain (e.g. finance).",
        task_type="generation",
        what_it_does="Reads a table/paragraph and works out an answer step by step — the FinQA / "
                     "financial-analysis style task. Scored on exact answer (execution accuracy).",
        honest_note="The hardest use case. Needs a bigger model AND a real dataset — this is the "
                    "'showcase' path: train on a public benchmark (FinQA/XBRL) and show a real gain "
                    "over the published baseline. Steers to Colab/cloud.",
        example_labels=[],
        example_rows=[
            {"input": "Revenue was 1,200 in 2021 and 1,500 in 2022. What was the % growth?", "target": "25%"},
        ],
        suggested_n=600, min_n=200, needs_bigger_model=True,
        local_model="Qwen/Qwen2.5-0.5B-Instruct", colab_model="Qwen/Qwen2.5-3B-Instruct",
        data_hint="Bring a reasoning dataset (or import a public one like FinQA) with question → exact answer.",
    ),
}


def get(key: str) -> Template | None:
    return TEMPLATES.get(key)


def as_dicts() -> list[dict]:
    return [asdict(t) for t in TEMPLATES.values()]
