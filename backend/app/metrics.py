"""Prometheus metrics exposed at /metrics."""

from prometheus_client import Counter, Histogram

REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "route", "status"])
REQUEST_LATENCY = Histogram("http_request_seconds", "HTTP request latency", ["route"])

STAGE_LATENCY = Histogram(
    "rag_stage_seconds", "Latency of each RAG stage", ["stage"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
CACHE_EVENTS = Counter("rag_cache_events_total", "Cache lookups", ["cache", "result"])
LLM_TOKENS = Counter("rag_llm_tokens_total", "AI model tokens", ["kind"])
INGESTED_CHUNKS = Counter("rag_ingested_chunks_total", "Chunks written by the ingestion worker")
INGEST_FAILURES = Counter("rag_ingest_failures_total", "Documents that failed ingestion")
