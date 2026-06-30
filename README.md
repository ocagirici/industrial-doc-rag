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

The harness runs the full RAG loop over a small Q/A set and reports two metrics:

- **Retrieval hit-rate** — did the gold source (and page, if specified) appear in
  the retrieved top-k chunks?
- **Keyword score** — fraction of expected keywords present in the answer (a cheap,
  API-free proxy for grounding; swappable for an LLM-as-judge).

```bash
python -m eval.run_eval     # writes eval/results.csv and eval/results.md
```

The cases in [`eval/eval_set.json`](eval/eval_set.json) are curated against a sample
corpus (a standing-desk equipment manual). Re-curate them for your own documents.
Because that corpus is a single document, hit-rate is matched at the **page** level
(source-level would be a trivial 100%).

### Results

`8 cases, top_k=5` — Retrieval hit-rate: **100%** · Mean keyword score: **100%**

| question | expected_source | hit | keyword_score |
| --- | --- | --- | --- |
| How do I reset the desk? | Desk User Manual.pdf | True | 1.0 |
| How do I set a memory height preset on the hand controller? | Desk User Manual.pdf | True | 1.0 |
| How do I lock the control panel display? | Desk User Manual.pdf | True | 1.0 |
| What is the difference between Jog and Continuation mode? | Desk User Manual.pdf | True | 1.0 |
| What is the USB port output voltage and current? | Desk User Manual.pdf | True | 1.0 |
| What height range can the maximum and minimum height limits be set to? | Desk User Manual.pdf | True | 1.0 |
| What is the tallest object I can place underneath the desk? | Desk User Manual.pdf | True | 1.0 |
| How long does assembly take and how many people are needed? | Desk User Manual.pdf | True | 1.0 |

> A clean sweep here reflects 8 answerable questions over one well-structured manual —
> a sanity-level smoke eval, not a stress test. Anti-hallucination is spot-checked
> separately: asking something the manual doesn't cover (e.g. "What is the warranty
> period?") correctly returns _"I don't know based on the provided documents."_

## What I'd add for production

This is a tight vertical slice. For a real deployment I'd add:

- **Auth & rate limiting** — API keys / OAuth on the FastAPI layer, per-client quotas.
- **Chat history** — persist conversations (e.g. MongoDB) for multi-turn context and audit.
- **Observability** — structured logging, request tracing, token/cost metrics, retrieval-quality dashboards.
- **Ingestion robustness** — OCR for scanned PDFs, table/figure extraction, dedup, background job queue for large uploads.
- **Retrieval quality** — hybrid (BM25 + vector) search, a reranker, and tuned chunking per document type.
- **Eval in CI** — run the harness on every change with an LLM-as-judge to catch regressions.
- **Deployment** — managed Postgres + pgvector, container orchestration (Azure Container Apps / K8s), secrets management, HTTPS.
