"""Retrieval: pgvector similarity, PostgreSQL full-text search, and hybrid fusion of the two."""

import re
from typing import Literal

import numpy as np

from . import cache, db
from .config import settings
from .ingest.embedding import get_embedder
from .metrics import STAGE_LATENCY
from .rerank import get_reranker, rerank

Mode = Literal["hybrid", "vector", "keyword"]
MODES: tuple[Mode, ...] = ("hybrid", "vector", "keyword")

_HIT_COLUMNS = """c.id AS chunk_id, c.document_id, d.title, d.filename, d.source_url,
                  c.page, c.heading, c.content"""


# Question words plus words that appear on almost every page of this corpus. OR-ing them into the
# keyword query floods the ranking with generic matches (PostgreSQL ts_rank has no IDF weighting).
STOPWORDS = frozenset("""
    a an and any are be can could do does for get how i if in is it me much many my of on or should the there
    to what when where which who will would you your someone something
    medicare cover covered covers coverage pay paid plan plans
""".split())


def rrf_merge(rankings: list[list[int]], k: int = 60, weights: list[float] | None = None) -> list[tuple[int, float]]:
    """Weighted reciprocal rank fusion: score(d) = sum of weight / (k + rank). Ranks start at 1.

    Fusing by rank (not raw score) works even though cosine similarity and ts_rank are on
    different scales, and a chunk found by both retrievers rises to the top.
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for weight, ranking in zip(weights, rankings):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def keyword_tsquery(text: str) -> str:
    """OR together the question's distinctive words so a natural-language question still matches.

    (websearch_to_tsquery ANDs every word, which rarely matches a whole question.)
    """
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in STOPWORDS]
    return " | ".join(dict.fromkeys(words))


def query_vector(text: str) -> np.ndarray:
    key = cache.query_embedding_key(text)
    cached = cache.get_json(key)
    if cached is None:
        with STAGE_LATENCY.labels("embed_query").time():
            cached = get_embedder().embed_query(text)
        cache.set_json(key, cached, ttl=7 * 24 * 3600)
    return np.array(cached, dtype=np.float32)


def _doc_filter(document_ids: list[int] | None) -> tuple[str, tuple]:
    if document_ids:
        return "AND d.id = ANY(%s)", (document_ids,)
    return "", ()


def vector_search(conn, text: str, limit: int, document_ids: list[int] | None = None) -> list[dict]:
    where, params = _doc_filter(document_ids)
    vec = query_vector(text)
    with STAGE_LATENCY.labels("vector_search").time():
        return conn.execute(
            f"""SELECT {_HIT_COLUMNS}, 1 - (c.embedding <=> %s) AS score
                FROM chunks c JOIN documents d ON d.id = c.document_id
                WHERE d.status = 'ready' {where}
                ORDER BY c.embedding <=> %s LIMIT %s""",
            (vec, *params, vec, limit),
        ).fetchall()


def keyword_search(conn, text: str, limit: int, document_ids: list[int] | None = None) -> list[dict]:
    tsq = keyword_tsquery(text)
    if not tsq:
        return []
    where, params = _doc_filter(document_ids)
    with STAGE_LATENCY.labels("keyword_search").time():
        return conn.execute(
            f"""SELECT {_HIT_COLUMNS}, ts_rank_cd(c.tsv, q, 1) AS score
                FROM chunks c JOIN documents d ON d.id = c.document_id, to_tsquery('english', %s) q
                WHERE d.status = 'ready' AND c.tsv @@ q {where}
                ORDER BY score DESC, c.id LIMIT %s""",
            (tsq, *params, limit),
        ).fetchall()


def search(text: str, mode: Mode | None = None, k: int | None = None,
           document_ids: list[int] | None = None, use_rerank: bool | None = None) -> list[dict]:
    """Return the top-k chunks for a query.

    Retrieves max(k, RERANK_POOL) candidates with the chosen mode (hybrid fuses 4x that many from each
    retriever), then lets the cross-encoder pick the best k when reranking is on.
    """
    mode = mode or settings.search_mode
    k = k or settings.top_k
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    use_rerank = (get_reranker() is not None) if use_rerank is None else use_rerank
    n = max(k, settings.rerank_pool) if use_rerank else k
    with db.connect() as conn:
        if mode == "vector":
            hits = vector_search(conn, text, n, document_ids)
        elif mode == "keyword":
            hits = keyword_search(conn, text, n, document_ids)
        else:
            vec_hits = vector_search(conn, text, n * 4, document_ids)
            kw_hits = keyword_search(conn, text, n * 4, document_ids)
            by_id = {h["chunk_id"]: h for h in vec_hits + kw_hits}
            fused = rrf_merge([[h["chunk_id"] for h in vec_hits], [h["chunk_id"] for h in kw_hits]],
                              settings.rrf_k, [1.0, settings.keyword_weight])
            hits = [{**by_id[chunk_id], "score": score} for chunk_id, score in fused[:n]]
    if use_rerank:
        hits = rerank(text, hits, k)
    return [{**h, "score": round(float(h["score"]), 6)} for h in hits[:k]]
