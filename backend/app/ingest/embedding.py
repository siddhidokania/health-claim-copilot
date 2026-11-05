"""Text embeddings. Runs a small local model (no API key, no per-call cost)."""

import hashlib
import math
import re
from functools import lru_cache
from typing import Protocol

from ..config import settings


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """BAAI/bge-small-en-v1.5 through fastembed (ONNX Runtime, CPU). bge models expect a query prefix."""

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str):
        import os

        from fastembed import TextEmbedding

        # Use every core: on a laptop CPU this is the difference between ~3 and ~5 chunks/s.
        self.model = TextEmbedding(model_name=model_name, threads=int(os.getenv("EMBED_THREADS", os.cpu_count() or 4)))
        self.dim = len(next(iter(self.model.embed(["probe"]))))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts, batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.embed([self.QUERY_PREFIX + text]))).tolist()


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder for tests: no model download, stable vectors."""

    def __init__(self, dim: int):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    if settings.embedding_model == "fake":
        return HashingEmbedder(settings.embedding_dim)
    embedder = FastEmbedder(settings.embedding_model)
    if embedder.dim != settings.embedding_dim:
        raise RuntimeError(f"{settings.embedding_model} returns {embedder.dim}-dim vectors but "
                           f"EMBEDDING_DIM={settings.embedding_dim}; update the setting and re-create the table")
    return embedder
