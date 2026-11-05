# Health Claim Copilot

[![CI](https://github.com/siddhidokania/health-claim-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/siddhidokania/health-claim-copilot/actions/workflows/ci.yml)

Health insurance claims in India often fail for reasons buried in a long policy document. This app reads your policy and explains it in plain language, with every answer linked to the exact clause and page.

## What it does

- Answers questions about your policy, with citations you can open and check.
- Reviews a refused claim: what the policy says, what to check, which documents to gather, and what is still unclear.
- Drafts a grievance letter to the insurer, using placeholders for personal details.
- Says so when the policy does not answer a question, instead of guessing.

This is general guidance, not legal, medical or financial advice. It never predicts whether a claim will be paid.

## How it works

1. A background worker splits the PDF into chunks and creates embeddings on your own computer.
2. Search finds the closest chunks with pgvector, then a reranker reorders them.
3. An AI model answers using only those chunks. A claim review searches every relevant topic first, then makes one model call.
4. Emails, phone numbers, PAN, Aadhaar, labelled policy or claim numbers and labelled names are removed before text is sent to the model.

Built with FastAPI, PostgreSQL with pgvector, Redis, fastembed and Docker Compose. Any OpenAI-compatible model works, including Gemini, Groq and Ollama, by changing `.env`.

## Run it

You need Docker Desktop.

```bash
cp .env.example .env
docker compose up --build
```

Add your key to `.env`, then open http://localhost:8000. Without a key, the app only shows the most relevant passages. To try it, upload `data/sample/sample-policy.txt`, a fictional policy. The first upload takes longer because the search models are downloaded.

Free AI tiers have daily request limits. If you reach one, wait until it resets or change `LLM_MODEL` in `.env`.

Use Gemini's free tier only with public documents, because free-tier content may be used by Google. For real documents, use Ollama so nothing leaves your computer.

## Privacy and limits

- Redaction uses patterns, so it can miss details. An unlabelled name, for example, is not removed. Do not treat it as complete.
- Everyone using one running copy can see the same documents. Run it only on your own computer until each user has separate, private storage.
- Deleting a policy in the app removes its file and stored text.

## Tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest tests/test_units.py tests/test_claims.py
```
