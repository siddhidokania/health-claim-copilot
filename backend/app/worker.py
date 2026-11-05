"""Background ingestion worker: pops document ids from the Redis queue and processes them.

Run with `python -m app.worker`. Several workers can run at once; each job is popped by exactly one,
and processing a document that is already 'ready' is a no-op, so duplicate queue entries are harmless.
"""

import logging
import signal
import time

import redis

from . import cache, db
from .ingest.embedding import get_embedder
from .ingest.pipeline import process_document
from .logging_setup import setup_logging

log = logging.getLogger("worker")
_running = True


def _stop(*_):
    global _running
    _running = False


def requeue_unfinished() -> None:
    """Documents left 'queued' or 'processing' (for example by a crashed worker) are queued again."""
    with db.connect() as conn:
        rows = conn.execute("SELECT id FROM documents WHERE status IN ('queued', 'processing') ORDER BY id").fetchall()
    for row in rows:
        cache.enqueue_ingest(row["id"])
    if rows:
        log.info("re-queued %d unfinished documents", len(rows))


def main() -> None:
    setup_logging()
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    db.init_schema()
    get_embedder()  # load the model once, before taking jobs
    requeue_unfinished()
    log.info("worker ready")
    while _running:
        try:
            doc_id = cache.next_ingest_job()
        except redis.RedisError as exc:
            log.warning("redis unavailable, retrying in 3s: %s", exc)
            time.sleep(3)
            continue
        if doc_id is not None:
            process_document(doc_id)
    db.close()
    log.info("worker stopped")


if __name__ == "__main__":
    main()
