"""FastAPI app: /health, /ingest, /ask.

Endpoints are async, but the ingestion and answering cores are synchronous
(embeddings, DB, LLM). We offload them with ``run_in_threadpool`` so a slow
embedding or LLM call never blocks the event loop — that's what makes the async
endpoints actually worth being async.

Run locally:
    uvicorn app.api.main:app --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.schemas import (
    AskRequest,
    AskResponse,
    FileIngestResult,
    IngestResponse,
    Source,
)
from app.core.db import init_db
from app.ingestion.pipeline import ingest_pdf
from app.retrieval.answer import answer_question

UPLOAD_DIR = Path("data/uploads")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure the schema exists before the app serves requests."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    yield


app = FastAPI(title="Industrial Document RAG Assistant", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile]) -> IngestResponse:
    """Upload one or more PDFs and run them through the ingestion pipeline."""
    results: list[FileIngestResult] = []
    for upload in files:
        if not (upload.filename or "").lower().endswith(".pdf"):
            raise HTTPException(400, f"not a PDF: {upload.filename!r}")

        dest = UPLOAD_DIR / Path(upload.filename).name
        dest.write_bytes(await upload.read())

        result = await run_in_threadpool(ingest_pdf, dest)
        results.append(
            FileIngestResult(
                source=result.source, pages=result.pages, chunks=result.chunks
            )
        )
    return IngestResponse(files=results)


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """Answer a question over the ingested documents, with source citations."""
    result = await run_in_threadpool(answer_question, request.question, request.top_k)
    return AskResponse(
        question=result.question,
        answer=result.answer,
        sources=[
            Source(
                source=chunk.source,
                page=chunk.page,
                chunk_index=chunk.chunk_index,
                score=chunk.score,
                content=chunk.content,
            )
            for chunk in result.sources
        ],
    )
