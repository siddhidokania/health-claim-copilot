"""Settings read from environment variables (see .env.example)."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_env_file() -> None:
    """Read the project's .env when running outside Docker. Real environment variables win."""
    path = Path(__file__).resolve().parents[2] / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


_load_env_file()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=lambda: os.getenv(
        "DATABASE_URL", "postgresql://rag:rag@localhost:5434/rag"))
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6380/0"))
    upload_dir: Path = field(default_factory=lambda: Path(os.getenv("UPLOAD_DIR", "data/uploads")))

    # Embeddings run locally (no API key needed); "fake" is a hashing embedder used by tests.
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    embedding_dim: int = field(default_factory=lambda: _int("EMBEDDING_DIM", 384))

    # Chunking
    chunk_chars: int = field(default_factory=lambda: _int("CHUNK_CHARS", 1000))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 150))

    # Retrieval. Defaults chosen from the retrieval eval (docs/evaluation.md).
    search_mode: str = field(default_factory=lambda: os.getenv("SEARCH_MODE", "vector"))
    top_k: int = field(default_factory=lambda: _int("TOP_K", 6))
    rrf_k: int = field(default_factory=lambda: _int("RRF_K", 60))
    keyword_weight: float = field(default_factory=lambda: float(os.getenv("KEYWORD_WEIGHT", "0.3")))
    rerank_model: str = field(default_factory=lambda: os.getenv("RERANK_MODEL", "jinaai/jina-reranker-v1-tiny-en"))
    rerank_pool: int = field(default_factory=lambda: _int("RERANK_POOL", 10))

    # AI model, through an OpenAI-compatible API. With no LLM_MODEL the app only shows passages.
    llm_base_url: str = field(default_factory=lambda: os.getenv(
        "LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    llm_max_tokens: int = field(default_factory=lambda: _int("LLM_MAX_TOKENS", 8000))

    # Cache
    answer_cache_ttl: int = field(default_factory=lambda: _int("ANSWER_CACHE_TTL", 24 * 3600))

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_model and (self.llm_api_key or "localhost" in self.llm_base_url))


settings = Settings()
