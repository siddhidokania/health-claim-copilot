"""FastAPI app: the web page, policy upload, claim review, streaming chat, health and metrics."""

import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from . import agent, cache, claims, db
from .config import settings
from .ingest.parsing import SUPPORTED
from .ingest.pipeline import process_document, register_document
from .logging_setup import request_id, setup_logging
from .metrics import REQUEST_LATENCY, REQUESTS
from .search import MODES, search

log = logging.getLogger("api")
MAX_UPLOAD_BYTES = 30 * 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    db.init_schema()
    log.info("api ready", extra={"llm": settings.llm_model if settings.llm_enabled else "extractive mode"})
    yield
    db.close()


app = FastAPI(title="Medicare RAG Assistant", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["*"],
                   allow_headers=["*"])


@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    token = request_id.set(rid)
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = rid
        return response
    finally:
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        REQUESTS.labels(request.method, path, status).inc()
        REQUEST_LATENCY.labels(path).observe(time.perf_counter() - started)
        if path not in ("/metrics", "/health"):
            log.info("request", extra={"method": request.method, "path": path, "status": status,
                                       "ms": round((time.perf_counter() - started) * 1000)})
        request_id.reset(token)


# --------------------------------------------------------------------------- health and metrics

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    checks = {}
    try:
        with db.connect() as conn:
            conn.execute("SELECT 1")
        checks["postgres"] = "ok"
    except Exception as exc:  # report, don't crash the probe
        checks["postgres"] = f"error: {type(exc).__name__}"
    try:
        cache.client().ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {type(exc).__name__}"
    checks["llm"] = settings.llm_model if settings.llm_enabled else "disabled (extractive mode)"
    return {"status": "ok" if checks["postgres"] == "ok" else "degraded", **checks}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# --------------------------------------------------------------------------- documents

def _safe_filename(name: str) -> str:
    stem, suffix = Path(name).stem, Path(name).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-.")[:80] or "document"
    return f"{stem}{suffix}"


@app.get("/api/documents")
def list_documents():
    with db.connect() as conn:
        return conn.execute(
            """SELECT id, title, filename, source_url, status, error, pages, chunks, created_at, updated_at
               FROM documents ORDER BY created_at DESC, id DESC""").fetchall()


@app.post("/api/documents", status_code=202)
async def upload_document(background: BackgroundTasks, file: UploadFile = File(...),
                          title: str | None = Query(None, max_length=200),
                          source_url: str | None = Query(None, max_length=500)):
    name = _safe_filename(file.filename or "document")
    if Path(name).suffix not in SUPPORTED:
        raise HTTPException(415, f"Unsupported file type; upload one of {sorted(SUPPORTED)}")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 30 MB")
    if not data:
        raise HTTPException(400, "File is empty")

    tmp = settings.upload_dir / f".upload-{uuid.uuid4().hex}"
    tmp.write_bytes(data)
    doc_id, created = register_document(tmp, title=title or Path(name).stem, source_url=source_url)
    final = settings.upload_dir / f"{doc_id}-{name}"
    if created:
        tmp.replace(final)
        with db.connect() as conn:
            conn.execute("UPDATE documents SET filename = %s WHERE id = %s", (final.name, doc_id))
        # The worker does the slow part (parse, embed); without Redis, fall back to an in-process task.
        if not cache.enqueue_ingest(doc_id):
            background.add_task(process_document, doc_id)
    else:
        tmp.unlink(missing_ok=True)
    return {"id": doc_id, "duplicate": not created}


@app.post("/api/documents/{doc_id}/reindex", status_code=202)
def reindex_document(doc_id: int, background: BackgroundTasks):
    with db.connect() as conn:
        row = conn.execute("UPDATE documents SET status = 'queued', updated_at = now() WHERE id = %s RETURNING id",
                           (doc_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Document not found")
    if not cache.enqueue_ingest(doc_id):
        background.add_task(process_document, doc_id, True)
    return {"id": doc_id}


@app.delete("/api/documents/{doc_id}", status_code=204)
def delete_document(doc_id: int):
    with db.connect() as conn:
        row = conn.execute("DELETE FROM documents WHERE id = %s RETURNING filename", (doc_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Document not found")
    (settings.upload_dir / row["filename"]).unlink(missing_ok=True)
    cache.bump_corpus_version()
    return Response(status_code=204)


# --------------------------------------------------------------------------- search and chat

@app.get("/api/search")
def search_chunks(q: str = Query(..., min_length=2, max_length=500), mode: str | None = None,
                  k: int = Query(6, ge=1, le=50), rerank: bool | None = None):
    mode = mode or settings.search_mode
    if mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")
    started = time.perf_counter()
    hits = search(q, mode, k, use_rerank=rerank)
    return {"query": q, "mode": mode, "latency_ms": round((time.perf_counter() - started) * 1000),
            "hits": [agent.source_payload(h) | {"score": h["score"]} for h in hits]}


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=20000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=20)
    mode: str | None = None
    document_ids: list[int] | None = None


def event_stream(events) -> StreamingResponse:
    rid = request_id.get()

    async def stream():
        request_id.set(rid)  # streaming runs after the middleware returns; keep the id on log lines
        try:
            async for event in events:
                yield agent.sse(event)
        except Exception:
            log.exception("chat failed")
            yield agent.sse({"type": "error", "message": "Something went wrong while answering."})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Server-sent events: sources, search, token, done or error. See agent.answer for the event shapes."""
    if req.mode is not None and req.mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")
    return event_stream(agent.answer(req.question, [t.model_dump() for t in req.history], req.mode, req.document_ids))


@app.post("/api/claim-review")
async def claim_review(req: claims.ClaimReview):
    """Same event stream as /api/chat, for a structured review of one claim against the policy."""
    return event_stream(agent.answer(claims.review_prompt(req), document_ids=req.document_ids, topics=claims.topics(req)))
