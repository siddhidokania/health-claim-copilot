"""Test setup: a hashing embedder (no model download) and a separate test database and Redis db.

Unit tests run anywhere. Tests marked `integration` need PostgreSQL with pgvector and Redis
(docker compose up -d db redis, or the service containers in CI) and are skipped otherwise.
"""

import os

os.environ["EMBEDDING_MODEL"] = "fake"
os.environ["RERANK_MODEL"] = "none"
os.environ["SEARCH_MODE"] = "hybrid"
os.environ["LLM_MODEL"] = ""
os.environ.setdefault("DATABASE_URL", "postgresql://rag:rag@localhost:5434/rag_test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6380/15")

import psycopg  # noqa: E402
import pytest  # noqa: E402
import redis  # noqa: E402


def _ensure_database(url: str) -> None:
    base, name = url.rsplit("/", 1)
    with psycopg.connect(f"{base}/postgres", autocommit=True, connect_timeout=3) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
            conn.execute(f'CREATE DATABASE "{name}"')


def _services_up() -> bool:
    try:
        _ensure_database(os.environ["DATABASE_URL"])
        redis.Redis.from_url(os.environ["REDIS_URL"], socket_connect_timeout=2).ping()
        return True
    except Exception:
        return False


SERVICES = _services_up()


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: needs PostgreSQL with pgvector and Redis")


def pytest_collection_modifyitems(items):
    if SERVICES:
        return
    skip = pytest.mark.skip(reason="PostgreSQL/Redis not reachable")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def clean_db(tmp_path):
    from app import cache, db
    from app.config import settings

    original = settings.upload_dir
    object.__setattr__(settings, "upload_dir", tmp_path)  # Settings is frozen; tests point uploads at tmp
    db.init_schema()
    with db.connect() as conn:
        conn.execute("TRUNCATE documents RESTART IDENTITY CASCADE")
    cache.client().flushdb()
    yield tmp_path
    object.__setattr__(settings, "upload_dir", original)


@pytest.fixture(scope="session", autouse=True)
def _close_pool():
    yield
    from app import db

    db.close()
