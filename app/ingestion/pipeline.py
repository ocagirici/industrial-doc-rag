"""Ingestion orchestration: PDF path -> rows in ``chunks``.

Ties the loader, splitter, and embedder together and persists the result. Run
directly to ingest a file from the command line and sanity-check the pipeline
before any API exists:

    python -m app.ingestion.pipeline path/to/manual.pdf
"""

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete

from app.core.db import Chunk, SessionLocal, init_db
from app.ingestion.embedder import get_embedder
from app.ingestion.loader import load_pdf
from app.ingestion.splitter import split_pages


@dataclass(frozen=True)
class IngestResult:
    """Summary of one ingest run, returned to callers (e.g. the API)."""

    source: str
    pages: int
    chunks: int


def ingest_pdf(path: str | Path) -> IngestResult:
    """Load, chunk, embed, and store one PDF.

    Re-ingesting a file with the same basename replaces its existing chunks, so
    repeated uploads stay idempotent instead of accumulating duplicates.
    """
    path = Path(path)
    pages = load_pdf(path)
    chunks = split_pages(pages)

    if not chunks:
        return IngestResult(source=path.name, pages=len(pages), chunks=0)

    embeddings = get_embedder().embed_documents([c.text for c in chunks])

    with SessionLocal.begin() as session:
        session.execute(delete(Chunk).where(Chunk.source == path.name))
        session.add_all(
            Chunk(
                source=c.source,
                page=c.page,
                chunk_index=c.chunk_index,
                content=c.text,
                embedding=vector,
            )
            for c, vector in zip(chunks, embeddings)
        )

    return IngestResult(source=path.name, pages=len(pages), chunks=len(chunks))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m app.ingestion.pipeline <path-to.pdf>")
        raise SystemExit(1)

    init_db()
    result = ingest_pdf(sys.argv[1])
    print(f"ingested {result.source}: {result.pages} pages -> {result.chunks} chunks")
