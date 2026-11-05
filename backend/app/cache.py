"""Redis: answer cache, query-embedding cache and the ingestion job queue.

Every call fails open: if Redis is down the app still answers, just without caching.
"""

import hashlib
import json
import logging
import re
from functools import lru_cache

import redis

from .config import settings
from .metrics import CACHE_EVENTS

log = logging.getLogger(__name__)

INGEST_QUEUE = "ingest:queue"
CORPUS_VERSION = "corpus:version"


@lru_cache(maxsize=1)
def client() -> redis.Redis:
    return redis.Redis.from_url(settings.redis_url, decode_responses=True, socket_timeout=2,
                                socket_connect_timeout=2)


def _normalise(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower().rstrip("?!. "))


def corpus_version() -> str:
    try:
        return client().get(CORPUS_VERSION) or "0"
    except redis.RedisError:
        return "0"


def bump_corpus_version() -> None:
    """Called after documents change so cached answers built on the old corpus stop matching."""
    try:
        client().incr(CORPUS_VERSION)
    except redis.RedisError as exc:
        log.warning("redis unavailable, corpus version not bumped: %s", exc)


def answer_key(question: str, mode: str) -> str:
    raw = f"{settings.llm_model}|{mode}|{corpus_version()}|{_normalise(question)}"
    return "answer:" + hashlib.sha256(raw.encode()).hexdigest()


def get_json(key: str) -> dict | list | None:
    try:
        value = client().get(key)
    except redis.RedisError:
        CACHE_EVENTS.labels(key.split(":")[0], "error").inc()
        return None
    CACHE_EVENTS.labels(key.split(":")[0], "hit" if value else "miss").inc()
    return json.loads(value) if value else None


def set_json(key: str, value, ttl: int) -> None:
    try:
        client().set(key, json.dumps(value), ex=ttl)
    except redis.RedisError as exc:
        log.warning("redis unavailable, not caching %s: %s", key.split(":")[0], exc)


def query_embedding_key(text: str) -> str:
    return f"qemb:{settings.embedding_model}:" + hashlib.sha256(_normalise(text).encode()).hexdigest()


def enqueue_ingest(document_id: int) -> bool:
    try:
        client().lpush(INGEST_QUEUE, document_id)
        return True
    except redis.RedisError as exc:
        log.warning("redis unavailable, cannot queue document %s: %s", document_id, exc)
        return False


def next_ingest_job(timeout: int = 1) -> int | None:
    # Must stay below the client socket_timeout (2s), or an idle blocking pop raises TimeoutError.
    item = client().brpop(INGEST_QUEUE, timeout=timeout)
    return int(item[1]) if item else None
