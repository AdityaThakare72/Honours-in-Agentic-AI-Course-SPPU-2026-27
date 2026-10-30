"""Shared helpers for Experiment 5: the SciFact benchmark and retrieval metrics.

SciFact (BEIR): 5,183 scientific abstracts (the corpus), 300 test claims (the queries), and for each claim the
abstract(s) that support or refute it (the relevance judgments, called qrels).
"""
import numpy as np
from datasets import load_dataset


def load_scifact(split="test"):
    corpus = load_dataset("mteb/scifact", "corpus")["corpus"]
    queries = load_dataset("mteb/scifact", "queries")["queries"]
    qrels = load_dataset("mteb/scifact", "default")[split]      # "test" (300 queries) or "train" (~800 queries)
    doc_ids = list(corpus["_id"])
    pos = {d: i for i, d in enumerate(doc_ids)}
    rel = {}
    for r in qrels:
        rel.setdefault(r["query-id"], set()).add(pos[r["corpus-id"]])
    q_text = dict(zip(queries["_id"], queries["text"]))
    q_ids = [q for q in q_text if q in rel]                      # only queries that have judgments
    return {
        "doc_ids": doc_ids,
        "titles": list(corpus["title"]),
        "texts": list(corpus["text"]),
        "q_ids": q_ids,
        "q_texts": [q_text[q] for q in q_ids],
        "relevant": [rel[q] for q in q_ids],                      # list of sets of document positions
    }


def evaluate(ranked, relevant, per_query=False):
    """ranked: (n_queries, >=100) array of document positions, best first. relevant: list of sets.
    Returns mean Recall@1/10/100, MRR@10 and nDCG@10 (binary relevance)."""
    rows = []
    discount = 1.0 / np.log2(np.arange(2, 12))                    # 1/log2(rank+1) for ranks 1..10
    for r, rel in zip(np.asarray(ranked), relevant):
        hit = np.array([d in rel for d in r[:100]], dtype=float)
        first = int(np.argmax(hit[:10])) if hit[:10].any() else None
        dcg = (hit[:10] * discount).sum()
        idcg = discount[: min(len(rel), 10)].sum()
        rows.append({"recall@1": hit[:1].sum() / len(rel), "recall@10": hit[:10].sum() / len(rel),
                     "recall@100": hit.sum() / len(rel),
                     "mrr@10": 0.0 if first is None else 1.0 / (first + 1), "ndcg@10": dcg / idcg})
    import pandas as pd
    df = pd.DataFrame(rows)
    return (df.mean().round(4), df) if per_query else df.mean().round(4)
