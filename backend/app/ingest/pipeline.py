"""Register uploaded files and turn them into embedded chunks."""

import hashlib
import logging
import time
from pathlib import Path

import numpy as np

from .. import cache, db
from ..config import settings
from ..metrics import INGEST_FAILURES, INGESTED_CHUNKS, STAGE_LATENCY
from .chunking import chunk_pages
from .embedding import get_embedder
from .parsing import extract_pages, title_from

log = logging.getLogger(__name__)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def register_document(path: Path, title: str | None = None, source_url: str | None = None) -> tuple[int, bool]:
    """Insert a queued document row. Returns (id, created); an identical file is not stored twice."""
    digest = sha256_of(path)
    with db.connect() as conn:
        row = conn.execute(
            """INSERT INTO documents (title, filename, sha256, source_url)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (sha256) DO NOTHING RETURNING id""",
            (title or path.stem, path.name, digest, source_url),
        ).fetchone()
        if row:
            return row["id"], True
        return conn.execute("SELECT id FROM documents WHERE sha256 = %s", (digest,)).fetchone()["id"], False


def _set_status(doc_id: int, status: str, **fields) -> None:
    sets = ", ".join(f"{k} = %({k})s" for k in fields)
    with db.connect() as conn:
        conn.execute(
            f"UPDATE documents SET status = %(status)s, updated_at = now(){', ' + sets if sets else ''} "
            "WHERE id = %(id)s",
            {"status": status, "id": doc_id, **fields},
        )


def process_document(doc_id: int, force: bool = False) -> int:
    """Parse, chunk, embed and store one document. Re-running replaces its chunks (idempotent);
    a document that is already 'ready' is skipped unless force=True (used for re-indexing)."""
    with db.connect() as conn:
        doc = conn.execute("SELECT * FROM documents WHERE id = %s", (doc_id,)).fetchone()
    if doc is None:
        log.warning("document %s no longer exists, skipping", doc_id)
        return 0
    if doc["status"] == "ready" and not force:
        return doc["chunks"] or 0
    path = settings.upload_dir / doc["filename"]
    _set_status(doc_id, "processing", error=None)
    started = time.perf_counter()
    try:
        with STAGE_LATENCY.labels("parse").time():
            pages = extract_pages(path)
        with STAGE_LATENCY.labels("chunk").time():
            chunks = chunk_pages(pages, settings.chunk_chars, settings.chunk_overlap)
        if not chunks:
            raise ValueError("no extractable text (is this a scanned PDF without a text layer?)")
        with STAGE_LATENCY.labels("embed_documents").time():
            vectors = get_embedder().embed_documents(
                [f"{c.heading}\n{c.content}" if c.heading else c.content for c in chunks])

        # A title that is just a file name (no spaces) is replaced by the document's own first heading.
        title = doc["title"] if " " in doc["title"] else title_from(path, pages)
        with db.connect() as conn, conn.transaction():
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO chunks (document_id, chunk_index, page, heading, content, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    [(doc_id, c.index, c.page, c.heading, c.content, np.array(v, dtype=np.float32))
                     for c, v in zip(chunks, vectors)],
                )
            conn.execute(
                """UPDATE documents SET status = 'ready', title = %s, pages = %s, chunks = %s,
                   error = NULL, updated_at = now() WHERE id = %s""",
                (title, len(pages), len(chunks), doc_id),
            )
        INGESTED_CHUNKS.inc(len(chunks))
        cache.bump_corpus_version()
        log.info("ingested document", extra={"document_id": doc_id, "pages": len(pages), "chunks": len(chunks),
                                             "seconds": round(time.perf_counter() - started, 2)})
        return len(chunks)
    except Exception as exc:
        INGEST_FAILURES.inc()
        log.exception("ingestion failed for document %s", doc_id)
        _set_status(doc_id, "failed", error=str(exc)[:500])
        return 0
