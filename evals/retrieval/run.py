"""Standalone retrieval eval.

Embeds a small in-memory corpus and each query with OpenAI embeddings, ranks by
cosine similarity, and scores recall@k / MRR / hit-rate against ground truth.

The production suite runs identical metrics over a live vector index; this
version needs only an embeddings key, so it runs anywhere.

Usage (from repo root):
    python -m evals.retrieval.run                 # default top_k
    python -m evals.retrieval.run --top-k 3
    python -m evals.retrieval.run --case attention
    python -m evals.retrieval.run --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from ..common import get_client, _git_sha
from .dataset import CORPUS, CASES

EMBED_MODEL = os.getenv("EVAL_EMBED_MODEL", "text-embedding-3-small")
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


async def _embed(texts: list[str]) -> list[list[float]]:
    resp = await get_client().embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _score_case(case: dict, ranked: list[tuple[dict, float]]) -> dict:
    """Compute retrieval metrics for one case against its ground truth.

    `ranked` is the top-k [(doc, score), ...] in rank order.
    """
    ranked_ids = [doc["resource_id"] for doc, _ in ranked]
    expected = case.get("expected_resource_ids") or []

    first_rank = None
    found = []
    for rid in expected:
        if rid in ranked_ids:
            found.append(rid)
            rank = ranked_ids.index(rid) + 1
            if first_rank is None or rank < first_rank:
                first_rank = rank

    return {
        "name": case["name"],
        "n_hits": len(ranked),
        "top_score": ranked[0][1] if ranked else None,
        "first_rank": first_rank,
        "recall": (len(found) / len(expected)) if expected else None,
        "rr": (1.0 / first_rank) if first_rank else 0.0,
        "hit": bool(found) if expected else None,
        "ranked": ranked,
    }


def _mean(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _fmt(x, pct=False):
    if x is None:
        return "n/a"
    return f"{x * 100:.0f}%" if pct else f"{x:.2f}"


async def run(top_k: int, filter_str: str | None, verbose: bool):
    """Embed the corpus and queries, rank by cosine, and report recall@k / MRR / hit-rate."""
    cases = CASES
    if filter_str:
        cases = [c for c in cases if filter_str.lower() in c["name"].lower()]
    if not cases:
        print(f"No cases matching '{filter_str}'")
        return

    print(f"Running {len(cases)} retrieval cases over {len(CORPUS)} docs (top_k={top_k})")
    print(f"embed model: {EMBED_MODEL}")
    print("=" * 70)

    start = time.time()
    corpus_vecs = await _embed([d["body"] for d in CORPUS])
    query_vecs = await _embed([c["query"] for c in cases])

    rows = []
    for case, qv in zip(cases, query_vecs):
        ranked = sorted(
            ((doc, _cosine(qv, cv)) for doc, cv in zip(CORPUS, corpus_vecs)),
            key=lambda t: t[1],
            reverse=True,
        )[:top_k]
        row = _score_case(case, ranked)
        rows.append(row)

        mark = {True: "✓", False: "✗", None: "·"}[row["hit"]]
        rank = f"@{row['first_rank']}" if row["first_rank"] else "miss"
        print(f"  {mark} {case['name']:<28} top={_fmt(row['top_score'])} {rank}")
        if verbose:
            for i, (doc, score) in enumerate(ranked):
                print(f"        {i + 1}. [{_fmt(score)}] {doc['title']} ({doc['resource_id']})")
            print()

    elapsed = time.time() - start
    recall = _mean([r["recall"] for r in rows])
    mrr = _mean([r["rr"] for r in rows if r["recall"] is not None])
    hit_rate = _mean([1.0 if r["hit"] else 0.0 for r in rows if r["hit"] is not None])
    mean_top = _mean([r["top_score"] for r in rows])

    print("=" * 70)
    print(
        f"recall@{top_k}: {_fmt(recall, pct=True)} | MRR: {_fmt(mrr)} | "
        f"hit rate: {_fmt(hit_rate, pct=True)} | mean top: {_fmt(mean_top)} | {elapsed:.1f}s"
    )

    path = _write_results(rows, top_k, recall, mrr, hit_rate, mean_top, elapsed)
    print(f"\nSaved: {path}")


def _write_results(rows, top_k, recall, mrr, hit_rate, mean_top, elapsed) -> Path:
    """Append a run entry to the top of results/retrieval.md (reverse-chrono)."""
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = [
        f"## {ts} · `{_git_sha()}`",
        "",
        f"- Params: `top_k={top_k}`, embed `{EMBED_MODEL}`",
        f"- **recall@{top_k}: {_fmt(recall, pct=True)} | MRR: {_fmt(mrr)} | "
        f"hit rate: {_fmt(hit_rate, pct=True)}**",
        f"- mean top score: {_fmt(mean_top)} | {len(rows)} cases | {elapsed:.1f}s",
        "",
        "| case | top | rank |",
        "| --- | --- | --- |",
    ]
    for r in rows:
        rank = r["first_rank"] if r["first_rank"] else "miss"
        entry.append(f"| {r['name']} | {_fmt(r['top_score'])} | {rank} |")
    entry += ["", "---", ""]
    entry_text = "\n".join(entry)

    md_path = RESULTS_DIR / "retrieval.md"
    header = (
        "# retrieval eval history\n\n"
        "Reverse-chronological. Each entry is one run; latest at the top.\n\n"
        "---\n\n"
    )
    existing = md_path.read_text() if md_path.exists() else header
    if not existing.startswith("# retrieval eval history"):
        existing = header + existing
    head, sep, body = existing.partition("---\n\n")
    md_path.write_text(head + sep + entry_text + body if sep else header + entry_text)
    return md_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run retrieval evals over an in-memory corpus")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--case", type=str, help="Filter cases by name substring")
    parser.add_argument("--verbose", action="store_true", help="Show every hit per case")
    args = parser.parse_args()
    asyncio.run(run(args.top_k, args.case, args.verbose))


if __name__ == "__main__":
    main()
