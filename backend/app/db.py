"""PostgreSQL + pgvector: connection pool and schema."""

from contextlib import contextmanager

from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import settings

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    filename    TEXT NOT NULL,
    sha256      TEXT NOT NULL UNIQUE,
    source_url  TEXT,
    status      TEXT NOT NULL DEFAULT 'queued'
                CHECK (status IN ('queued', 'processing', 'ready', 'failed')),
    error       TEXT,
    pages       INT,
    chunks      INT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL,
    page         INT NOT NULL,
    heading      TEXT,
    content      TEXT NOT NULL,
    embedding    vector({dim}) NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(heading, '') || ' ' || content)) STORED,
    UNIQUE (document_id, chunk_index)
);

-- Approximate nearest-neighbour index for cosine similarity, and a GIN index for keyword search.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_document ON chunks (document_id);
"""

_pool: ConnectionPool | None = None


def _configure(conn):
    register_vector(conn)


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(settings.database_url, min_size=1, max_size=10, open=True,
                               kwargs={"row_factory": dict_row}, configure=_configure)
    return _pool


@contextmanager
def connect():
    with pool().connection() as conn:
        yield conn


def init_schema() -> None:
    # The extension must exist before register_vector can find the type, so use a plain connection first.
    import psycopg

    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        conn.execute(SCHEMA.replace("{dim}", str(settings.embedding_dim)))


def close() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
