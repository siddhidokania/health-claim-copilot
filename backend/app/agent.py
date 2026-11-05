"""Answer questions from retrieved policy text, streaming events to the client.

Flow for one question:
  1. Search for the question, plus any extra topics (a claim review searches several).
  2. Send those excerpts and the question to the AI model in a single call.
  3. The model replies in one piece, citing excerpts as [S<id>]; the client turns those into links.

The model is reached through the OpenAI-compatible API that Gemini, Groq and Ollama all offer, so
LLM_BASE_URL and LLM_MODEL in .env choose the provider. Without a model configured the app runs in
extractive mode: it returns the best passages with the same citation format.
"""

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator

import httpx

from .config import settings
from .metrics import LLM_TOKENS, STAGE_LATENCY
from .privacy import redact
from .search import search

log = logging.getLogger(__name__)

CITATION = re.compile(r"\[S(\d+)\]")
MAX_SOURCES = 14
ERRORS = {
    400: "The AI service rejected the request. Check LLM_MODEL in .env.",
    401: "The API key was rejected. Check LLM_API_KEY in .env.",
    403: "The API key was rejected. Check LLM_API_KEY in .env.",
    404: "That AI model was not found. Check LLM_MODEL in .env.",
    429: "The free AI limit has been reached. Wait a minute and try again.",
    503: "The AI service is busy right now. Wait a minute and try again.",
}
DAILY_LIMIT = "Today's free AI limit has been used up. Try again tomorrow, or switch LLM_MODEL in .env."
BUSY_RETRIES = 3
RETRY_WAIT = 3  # seconds, multiplied by the attempt number

SYSTEM_PROMPT = """You help people in India understand their health insurance policy and refused claims. You use only \
excerpts that this app retrieved from the documents they uploaded.

Each excerpt is labelled with a source id such as [S123]. Rules:
- Base every statement about the policy on the excerpts. Cite the source id in square brackets right after the \
statement it supports, for example "Cataract surgery has a 24-month waiting period [S123]." Cite only ids you were given, \
each in its own brackets.
- If the excerpts do not contain the answer, say so plainly and name who could answer it, such as the insurer, the TPA or \
the hospital, without citing anything. Never fill gaps from general knowledge, and never assume one insurer's terms apply to another.
- Quote amounts, percentages, time limits and waiting periods exactly as the excerpt states them.
- Explain insurance words in simple language the first time you use them.
- Never promise an outcome. Do not say a claim will be paid, or that an insurer acted wrongly. Point out what the person \
can check or ask.
- Personal details appear as placeholders like [NAME] or [NUMBER]. Keep them as placeholders.
- This is general information, not legal, medical or financial advice."""


def http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=httpx.Timeout(90, connect=15))


def format_hits(hits: list[dict]) -> str:
    if not hits:
        return "No matching passages."
    blocks = []
    for h in hits:
        where = f"{h['title']}, page {h['page']}" + (f" - {h['heading']}" if h.get("heading") else "")
        blocks.append(f"[S{h['chunk_id']}] {where}\n{h['content']}")
    return redact("\n\n".join(blocks))  # everything from documents passes through here on its way to the model


def source_payload(h: dict) -> dict:
    return {"id": h["chunk_id"], "document_id": h["document_id"], "title": h["title"], "page": h["page"],
            "heading": h.get("heading"), "content": h["content"], "source_url": h.get("source_url")}


def cited_ids(text: str, known: set[int]) -> list[int]:
    """Source ids cited in the answer, in first-cited order, ignoring any id we never supplied."""
    seen: dict[int, None] = {}
    for m in CITATION.finditer(text):
        cid = int(m.group(1))
        if cid in known:
            seen.setdefault(cid, None)
    return list(seen)


def _extractive_answer(hits: list[dict]) -> str:
    if not hits:
        return "I couldn't find anything about that in the loaded documents."
    lines = ["Search-only mode (no AI model set up): these are the most relevant passages.\n"]
    for h in hits[:3]:
        snippet = h["content"].replace("\n", " ")
        snippet = snippet[:400] + ("..." if len(snippet) > 400 else "")
        lines.append(f"- {snippet} [S{h['chunk_id']}]")
    return "\n".join(lines)


def retrieve(question: str, topics: list[str], mode: str, document_ids: list[int] | None) -> list[dict]:
    """Search once per topic and keep each passage once, so one model call can cover a whole claim."""
    found: dict[int, dict] = {}
    for hit in search(question, mode, settings.top_k, document_ids):
        found[hit["chunk_id"]] = hit
    for topic in topics:
        for hit in search(topic, mode, 3, document_ids):
            found.setdefault(hit["chunk_id"], hit)
    return list(found.values())[:MAX_SOURCES]


async def call_model(messages: list[dict]) -> dict:
    """Return the model's whole reply. Asking for it in one piece means a provider failure can't leave half an answer."""
    body = {"model": settings.llm_model, "messages": messages, "temperature": 0.1, "max_tokens": settings.llm_max_tokens}
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
    async with http_client() as client:
        for attempt in range(BUSY_RETRIES + 1):
            response = await client.post(f"{settings.llm_base_url.rstrip('/')}/chat/completions", json=body, headers=headers)
            if response.status_code == 503 and attempt < BUSY_RETRIES:
                await asyncio.sleep(RETRY_WAIT * (attempt + 1))
                continue
            if response.status_code != 200:
                log.error("model call failed", extra={"status": response.status_code, "body": response.text[:300]})
                raise httpx.HTTPStatusError("model call failed", request=response.request, response=response)
            return response.json()
    raise AssertionError("unreachable")


async def answer(question: str, history: list[dict] | None = None, mode: str | None = None,
                 document_ids: list[int] | None = None, topics: list[str] | None = None) -> AsyncIterator[dict]:
    """Yield events: sources, token, done (or error). See main.py for the SSE wire format."""
    started = time.perf_counter()
    mode = mode or settings.search_mode
    history = [m for m in (history or []) if m.get("role") in ("user", "assistant") and m.get("content")][-6:]
    question = redact(question)
    history = [{"role": m["role"], "content": redact(m["content"])} for m in history]

    with STAGE_LATENCY.labels("retrieve").time():
        hits = await asyncio.to_thread(retrieve, question, topics or [], mode, document_ids)
    sources = {h["chunk_id"]: h for h in hits}
    yield {"type": "sources", "sources": [source_payload(h) for h in hits]}

    if not settings.llm_enabled:
        text = _extractive_answer(hits)
        yield {"type": "token", "text": text}
        yield {"type": "done", "cited": cited_ids(text, set(sources)), "cached": False, "mode": "extractive",
               "latency_ms": round((time.perf_counter() - started) * 1000), "usage": {}}
        return

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history,
                {"role": "user", "content": f"Question: {question}\n\nExcerpts from the policy:\n\n{format_hits(hits)}"}]
    try:
        with STAGE_LATENCY.labels("llm_turn").time():
            reply = await call_model(messages)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429 and "per day" in exc.response.text.lower():
            yield {"type": "error", "message": DAILY_LIMIT}
            return
        yield {"type": "error", "message": ERRORS.get(exc.response.status_code, "The AI service had a problem. Try again.")}
        return
    except httpx.TimeoutException:
        log.exception("model call timed out")
        yield {"type": "error", "message": "The AI service took too long to reply. Try again."}
        return
    except httpx.HTTPError:
        log.exception("model call failed")
        yield {"type": "error", "message": "Could not reach the AI service. Check your internet connection."}
        return

    choice = (reply.get("choices") or [{}])[0]
    text = ((choice.get("message") or {}).get("content") or "").strip()
    usage = reply.get("usage") or {}
    if not text:
        yield {"type": "error", "message": "The AI service returned no answer. Try rewording your question."}
        return
    if choice.get("finish_reason") != "stop":
        log.warning("answer incomplete", extra={"finish": choice.get("finish_reason")})
        yield {"type": "error", "message": "The answer was cut off. Please ask again."}
        return
    yield {"type": "token", "text": text}
    for name in ("prompt_tokens", "completion_tokens"):
        LLM_TOKENS.labels(name).inc(usage.get(name, 0) or 0)
    cited = cited_ids(text, set(sources))
    latency = round((time.perf_counter() - started) * 1000)
    log.info("answered", extra={"latency_ms": latency, "cited": len(cited), "sources": len(sources)})
    yield {"type": "done", "cited": cited, "cached": False, "mode": settings.llm_model, "latency_ms": latency, "usage": usage}


def sse(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"
