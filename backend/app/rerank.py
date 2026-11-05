"""Cross-encoder reranking: re-score the top candidates by reading the question and passage together.

The embedding model compares a question vector with passage vectors computed separately; a cross-encoder
reads both at once, which is slower but more precise. Running it on only the top RERANK_POOL candidates
keeps the cost to a few hundred milliseconds on CPU.
"""

from functools import lru_cache

from .config import settings
from .metrics import STAGE_LATENCY


@lru_cache(maxsize=1)
def get_reranker():
    if settings.rerank_model in ("", "none"):
        return None
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=settings.rerank_model)


def rerank(query: str, hits: list[dict], k: int) -> list[dict]:
    model = get_reranker()
    if model is None or len(hits) <= 1:
        return hits[:k]
    with STAGE_LATENCY.labels("rerank").time():
        scores = list(model.rerank(query, [f"{h.get('heading') or ''}\n{h['content']}" for h in hits]))
    ranked = sorted(zip(scores, hits), key=lambda pair: -pair[0])
    return [{**h, "score": float(score)} for score, h in ranked[:k]]
