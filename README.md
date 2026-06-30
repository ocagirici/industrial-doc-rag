# Industrial Document RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant for querying industrial documents
(equipment manuals, datasheets, maintenance guides). Upload PDFs, ask questions in
natural language, and get grounded answers with source citations (file + page).

> Personal portfolio project demonstrating production-style GenAI engineering:
> Python, FastAPI, RAG, LLM integration, pgvector, and a small eval harness.

## Architecture

```
                          ┌─────────────────────────────────────────────┐
   PDF upload ──ingest──▶ │ load → chunk → embed → store (pgvector)      │
                          └─────────────────────────────────────────────┘
                                              │
                                  Postgres + pgvector (HNSW)
                                              │
   question ──▶ embed ──▶ top-k cosine retrieve ──▶ context+prompt ──▶ LLM ──▶ grounded answer + sources
                                              │
                          FastAPI (REST)  ◀── Streamlit (chat UI)
                                              │
                          eval harness (retrieval hit-rate + answer quality)
```

**Default stack (cost-conscious):** local `all-MiniLM-L6-v2` embeddings (384-dim,
no API cost) + an API LLM (Claude Haiku 4.5 by default) for generation. Both the
embedding and LLM providers are swappable via env vars — see `.env.example`.

LangChain is used only for the ingestion plumbing (PDF loader + recursive token
splitter). The core `retrieve → prompt → generate` loop is plain, readable Python
in [`app/retrieval/`](app/retrieval/) — no chains hiding the logic.

## Project layout

```
app/
  core/        config (typed settings) + db (pgvector schema)
  ingestion/   loader → splitter → embedder → pipeline
  retrieval/   retriever → prompt → generator → answer (the RAG loop)
  api/         FastAPI app + schemas
ui/            Streamlit chat client
eval/          eval set + harness
```

## Status

- [x] 1. Ingestion pipeline (load → chunk → embed → store)
- [x] 2. Retrieval + generation (retrieve → prompt → LLM)
- [x] 3. FastAPI backend (`/ingest`, `/ask`, `/health`)
- [x] 4. Streamlit UI
- [x] 5. Eval harness
- [x] 6. Docker / docker-compose
- [x] 7. Docs

## Setup & run (docker-compose)

```bash
cp .env.example .env          # set ANTHROPIC_API_KEY (or switch providers)
docker compose up --build
```

This starts three services:

| Service | URL | What it is |
|---------|-----|------------|
| `db`  | `localhost:5432` | Postgres + pgvector |
| `api` | `localhost:8000` | FastAPI (`/docs` for Swagger UI) |
| `ui`  | `localhost:8501` | Streamlit chat interface |

Open the UI at **http://localhost:8501**, upload PDF(s) in the sidebar, click
**Ingest**, then ask questions.

### Running locally (without Docker)

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
# point DATABASE_URL at a local pgvector Postgres, then:
uvicorn app.api.main:app --reload                       # API on :8000
streamlit run ui/streamlit_app.py                       # UI on :8501
```

CLI smoke tests (handy before the API/UI are up):

```bash
python -m app.ingestion.pipeline path/to/manual.pdf
python -m app.retrieval.answer "What is the maximum operating pressure?"
```

## Example query / response

```
POST /ask   { "question": "What torque should the housing bolts be tightened to?" }

{
  "question": "What torque should the housing bolts be tightened to?",
  "answer": "Tighten the housing bolts to 40 Nm in a crosswise pattern [pump_manual.pdf p.5].",
  "sources": [
    { "source": "pump_manual.pdf", "page": 5, "chunk_index": 12, "score": 0.83,
      "content": "Housing bolts must be tightened to 40 Nm using a crosswise sequence..." }
  ]
}
```

If the retrieved context doesn't contain the answer, the assistant replies
`I don't know based on the provided documents.` rather than guessing
(anti-hallucination grounding enforced by the system prompt in
[`app/retrieval/prompt.py`](app/retrieval/prompt.py)).

## Evaluation

A constraint-based harness (inspired by
[agent-eval-harness](https://github.com/ocagirici/agent-eval-harness)) scores each
case along two axes:

- **Answer pass/fail** — the answer is checked against per-case *constraints*, not a
  fuzzy keyword score, so the eval can actually fail and discriminate.
- **Retrieval hit-rate** — page-level diagnostic: did the gold page appear in the
  retrieved top-k? (Source-level would be a trivial 100% for a single-doc corpus.)

Each case in [`eval/testset.json`](eval/testset.json) combines any of these
constraints (full schema in [`eval/evaluator.py`](eval/evaluator.py)):

| field | meaning |
|-------|---------|
| `expected` | every term must appear (AND) |
| `expected_any` | at least one must appear (OR) |
| `expected_exactly` + `candidate_pool` | these and nothing else from the pool |
| `expected_none` | none may appear — catches over-answering / hallucination |
| `must_refuse` | answer must be the "I don't know" refusal (adversarial / unanswerable) |
| `expected_page` | gold page for the retrieval check |

What gives it teeth: **adversarial trap cases** (unanswerable questions that must be
refused; over-answer baits), **token-boundary numeric matching** (so `5` matches
`5V` but not `0.85` or `120`), and **failure categorization**
(`retrieval_miss` / `missing_fact` / `over_answer` / `hallucination`).

**Add your own cases** by editing `testset.json`, or point at a different file:

```bash
python -m eval.run_eval                          # default testset, top_k=5
python -m eval.run_eval --testset my_cases.json --top-k 8
# in Docker: docker compose exec api python -m eval.run_eval
```

### Results

`12 cases, top_k=5` — **Answer pass-rate: 12/12 (100%)** · Retrieval hit-rate: **100%**

| id | difficulty | retrieval_hit | passed | category |
| --- | --- | --- | --- | --- |
| easy-usb-current | easy | True | True | pass |
| easy-object-height | easy | True | True | pass |
| easy-assembly-people | easy | True | True | pass |
| med-reset | medium | True | True | pass |
| med-lock | medium | True | True | pass |
| med-height-limit | medium | True | True | pass |
| med-jog-continuation | medium | True | True | pass |
| trap-lower-key | hard | True | True | pass |
| trap-warranty | trap | — | True | pass (refused) |
| trap-price | trap | — | True | pass (refused) |
| trap-bluetooth | trap | — | True | pass (refused) |
| trap-usb-voltage-only | hard | True | True | pass |

The harness is **not** pinned at 100% — it discriminates. Starving retrieval to
`--top-k 1` drops it to **10/12 (83%)** with retrieval-induced failures, and earlier
trap iterations surfaced `hallucination`-category failures. A clean sweep at `top_k=5`
means Claude Haiku genuinely handles these 12 cases — including 3 unanswerable
refusals and 2 over-answer/numeric traps — not that the metric is blunt.

## What I'd add for production

This is a tight vertical slice. For a real deployment I'd add:

- **Auth & rate limiting** — API keys / OAuth on the FastAPI layer, per-client quotas.
- **Chat history** — persist conversations (e.g. MongoDB) for multi-turn context and audit.
- **Observability** — structured logging, request tracing, token/cost metrics, retrieval-quality dashboards.
- **Ingestion robustness** — OCR for scanned PDFs, table/figure extraction, dedup, background job queue for large uploads.
- **Retrieval quality** — hybrid (BM25 + vector) search, a reranker, and tuned chunking per document type.
- **Eval in CI** — run the harness on every change with an LLM-as-judge to catch regressions.
- **Deployment** — managed Postgres + pgvector, container orchestration (Azure Container Apps / K8s), secrets management, HTTPS.
