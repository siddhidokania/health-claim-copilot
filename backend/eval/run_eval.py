"""Retrieval evaluation: does search return a passage that contains the answer?

Each question in questions.jsonl has an `answer_phrase` copied from the passage that answers it.
A retrieved chunk is relevant if it contains that phrase (after normalising quotes, dashes and
spaces), so the metric does not depend on how the documents were chunked.

    python -m eval.run_eval              # from backend/, with the corpus ingested

Writes eval/results.json and prints a Markdown table (recall@1/3/6, MRR@10, latency) per mode.
"""

import json
import re
import statistics
import sys
import time
from pathlib import Path

from app import db
from app.search import search

HERE = Path(__file__).parent
K_VALUES = (1, 3, 6)
DEPTH = 10


def norm(text: str) -> str:
    text = text.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                                         "–": "-", "—": "-", " ": " "}))
    return re.sub(r"\s+", " ", text).lower()


def load_questions() -> list[dict]:
    return [json.loads(line) for line in (HERE / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line]


def check_phrases(questions: list[dict]) -> None:
    """Every answer phrase must exist in the corpus, or the question can never be answered."""
    with db.connect() as conn:
        contents = [norm(r["content"]) for r in conn.execute("SELECT content FROM chunks").fetchall()]
    missing = [q["id"] for q in questions if not any(norm(q["answer_phrase"]) in c for c in contents)]
    if missing:
        sys.exit(f"answer phrase not found in any chunk for question(s) {missing}; fix questions.jsonl")


# (label, mode, rerank). Reranking reorders the top RERANK_POOL candidates, so it is scored to depth 10 too.
CONFIGS = [("keyword", "keyword", False), ("hybrid", "hybrid", False), ("vector", "vector", False),
           ("hybrid + rerank", "hybrid", True), ("vector + rerank", "vector", True)]


def evaluate(questions: list[dict], label: str, mode: str, use_rerank: bool) -> dict:
    ranks, latencies, misses = [], [], []
    for q in questions:
        phrase = norm(q["answer_phrase"])
        started = time.perf_counter()
        hits = search(q["question"], mode, DEPTH, use_rerank=use_rerank)
        latencies.append((time.perf_counter() - started) * 1000)
        rank = next((i for i, h in enumerate(hits, start=1) if phrase in norm(h["content"])), None)
        ranks.append(rank)
        if rank is None or rank > max(K_VALUES):
            misses.append(q["id"])
    n = len(questions)
    return {
        "mode": label,
        **{f"recall@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 3) for k in K_VALUES},
        f"mrr@{DEPTH}": round(sum(1 / r for r in ranks if r) / n, 3),
        "p50_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(sorted(latencies)[int(0.95 * (n - 1))], 1),
        "missed_at_6": misses,
    }


def main() -> None:
    questions = load_questions()
    check_phrases(questions)
    for q in questions[:3]:  # warm up the embedding model and caches so latency is steady-state
        search(q["question"], "vector", 1)
    results = [evaluate(questions, mode) for mode in MODES]
    (HERE / "results.json").write_text(json.dumps({"questions": len(questions), "results": results}, indent=2))

    cols = ["recall@1", "recall@3", "recall@6", f"mrr@{DEPTH}", "p50_ms", "p95_ms"]
    print(f"Retrieval eval: {len(questions)} questions\n")
    print("| mode | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for r in results:
        print(f"| {r['mode']} | " + " | ".join(str(r[c]) for c in cols) + " |")
    for r in results:
        if r["missed_at_6"]:
            print(f"\n{r['mode']} missed (not in top 6): questions {r['missed_at_6']}")
    db.close()


if __name__ == "__main__":
    main()
