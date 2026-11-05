"""End-to-end tests against PostgreSQL + pgvector and Redis, with the hashing embedder."""

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import agent, cache
from app.ingest.pipeline import process_document, register_document
from app.search import search

pytestmark = pytest.mark.integration

HOSPICE = """Medicare Hospice Benefits

Who is eligible for hospice care?
You can get hospice care if you have Medicare Part A and your doctor certifies that you are
terminally ill, with a life expectancy of 6 months or less.

What hospice covers
Hospice covers pain relief, nursing care, medical equipment and short-term respite care
for your caregiver.
"""

DIABETES = """Medicare Coverage of Diabetes Supplies

Blood sugar test strips
Part B covers blood sugar self-testing equipment and supplies, such as test strips and lancets,
as durable medical equipment.

Insulin pumps
Part B covers external insulin pumps and the insulin used with them.
"""


def add_doc(folder, name, text, process=True) -> int:
    path = folder / name
    path.write_text(text, encoding="utf-8")
    doc_id, created = register_document(path, title=name.removesuffix(".txt").replace("-", " ").title())
    assert created
    if process:
        assert process_document(doc_id) > 0
    return doc_id


def test_ingest_is_idempotent_and_deduplicates(clean_db):
    doc_id = add_doc(clean_db, "hospice.txt", HOSPICE)
    assert register_document(clean_db / "hospice.txt")[1] is False      # same bytes: no second row
    assert process_document(doc_id) == process_document(doc_id, force=True)


def test_search_modes_find_the_right_document(clean_db):
    add_doc(clean_db, "hospice.txt", HOSPICE)
    add_doc(clean_db, "diabetes.txt", DIABETES)
    for mode in ("hybrid", "vector", "keyword"):
        top = search("Does Part B cover test strips and lancets?", mode, 3)
        assert top and "Diabetes" in top[0]["title"], mode
    hospice = search("who is eligible for hospice", "hybrid", 2)
    assert hospice[0]["heading"] == "Who is eligible for hospice care?"


def test_failed_document_is_marked_failed(clean_db):
    doc_id = add_doc(clean_db, "empty.txt", "   \n\n  ", process=False)
    assert process_document(doc_id) == 0
    from app import db

    with db.connect() as conn:
        row = conn.execute("SELECT status, error FROM documents WHERE id = %s", (doc_id,)).fetchone()
    assert row["status"] == "failed" and "no extractable text" in row["error"]


def _events(body: str) -> list[dict]:
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


def test_api_upload_search_and_extractive_chat(clean_db, monkeypatch):
    from app.main import app

    # Run ingestion inline instead of through the worker queue.
    monkeypatch.setattr(cache, "enqueue_ingest", lambda _id: False)
    with TestClient(app) as client:
        assert client.get("/health").json()["postgres"] == "ok"
        r = client.post("/api/documents", files={"file": ("hospice-guide.txt", HOSPICE.encode(), "text/plain")})
        assert r.status_code == 202 and r.json()["duplicate"] is False
        again = client.post("/api/documents", files={"file": ("copy.txt", HOSPICE.encode(), "text/plain")})
        assert again.json() == {"id": r.json()["id"], "duplicate": True}
        assert client.post("/api/documents", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 415

        docs = client.get("/api/documents").json()
        assert docs[0]["status"] == "ready" and docs[0]["chunks"] >= 2

        hits = client.get("/api/search", params={"q": "respite care for caregiver"}).json()["hits"]
        assert "respite" in hits[0]["content"]

        body = client.post("/api/chat", json={"question": "What does hospice cover?"}).text
        events = _events(body)
        assert [e["type"] for e in events] == ["sources", "token", "done"]
        assert events[-1]["mode"] == "extractive" and events[-1]["cited"]

        cached = _events(client.post("/api/chat", json={"question": "What does hospice cover?"}).text)
        assert cached[-1]["cached"] is False  # answers can hold personal details, so they are never stored

        assert client.delete(f"/api/documents/{r.json()['id']}").status_code == 204
        assert client.get("/api/documents").json() == []
        assert "rag_stage_seconds" in client.get("/metrics").text


# ---------------------------------------------------------------- model call with a fake provider

def reply(text, finish="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def fake_provider(monkeypatch, handler):
    monkeypatch.setattr(agent, "http_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    object.__setattr__(agent.settings, "llm_model", "test-model")
    object.__setattr__(agent.settings, "llm_api_key", "test-key")


def reset_provider():
    object.__setattr__(agent.settings, "llm_model", "")
    object.__setattr__(agent.settings, "llm_api_key", "")


def run_answer(*args, **kwargs):
    async def run():
        return [e async for e in agent.answer(*args, **kwargs)]
    return asyncio.run(run())


def test_answer_searches_every_topic_and_cites_sources(clean_db, monkeypatch):
    add_doc(clean_db, "hospice.txt", HOSPICE)
    add_doc(clean_db, "diabetes.txt", DIABETES)
    pump_chunk = search("insulin pumps", "keyword", 1)[0]["chunk_id"]
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=reply(f"Pumps are covered [S{pump_chunk}]."))

    fake_provider(monkeypatch, handler)
    try:
        events = run_answer("Does the plan cover hospice?", mode="keyword", topics=["insulin pumps"])
    finally:
        reset_provider()

    assert len(calls) == 1 and calls[0]["model"] == "test-model"
    assert f"[S{pump_chunk}]" in calls[0]["messages"][-1]["content"]
    assert events[-1]["type"] == "done" and events[-1]["cited"] == [pump_chunk]


@pytest.mark.parametrize("status,phrase", [(429, "limit"), (401, "key"), (404, "model"), (503, "busy")])
def test_provider_errors_become_clear_messages(clean_db, monkeypatch, status, phrase):
    add_doc(clean_db, "hospice.txt", HOSPICE)
    monkeypatch.setattr(agent, "RETRY_WAIT", 0)
    fake_provider(monkeypatch, lambda request: httpx.Response(status, text="nope"))
    try:
        events = run_answer("What does hospice cover?")
    finally:
        reset_provider()
    assert events[-1]["type"] == "error" and phrase in events[-1]["message"]


def test_busy_provider_is_retried(clean_db, monkeypatch):
    add_doc(clean_db, "hospice.txt", HOSPICE)
    monkeypatch.setattr(agent, "RETRY_WAIT", 0)
    replies = [httpx.Response(503, text="busy"),
               httpx.Response(200, json=reply("Covered."))]
    fake_provider(monkeypatch, lambda request: replies.pop(0))
    try:
        events = run_answer("What does hospice cover?")
    finally:
        reset_provider()
    assert not replies and events[-1]["type"] == "done"


def test_cut_off_answer_is_not_shown(clean_db, monkeypatch):
    add_doc(clean_db, "hospice.txt", HOSPICE)
    cut_off = reply("The waiting period is", finish="length")
    fake_provider(monkeypatch, lambda request: httpx.Response(200, json=cut_off))
    try:
        events = run_answer("What does hospice cover?")
    finally:
        reset_provider()
    assert events[-1]["type"] == "error" and "cut off" in events[-1]["message"]
    assert not any(e["type"] == "token" for e in events)


def test_daily_limit_gets_its_own_message(clean_db, monkeypatch):
    add_doc(clean_db, "hospice.txt", HOSPICE)
    fake_provider(monkeypatch, lambda request: httpx.Response(429, text="Requests per day limit exceeded"))
    try:
        events = run_answer("What does hospice cover?")
    finally:
        reset_provider()
    assert events[-1]["type"] == "error" and "Today's free AI limit" in events[-1]["message"]
