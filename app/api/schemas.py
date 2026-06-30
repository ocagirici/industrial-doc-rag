"""Pydantic request/response models — the API's public contract.

Kept separate from the FastAPI app so the shapes are easy to find and reuse
(e.g. the Streamlit UI mirrors these fields).
"""

from pydantic import BaseModel, Field


class FileIngestResult(BaseModel):
    """Per-file outcome of an ingest run."""

    source: str
    pages: int
    chunks: int


class IngestResponse(BaseModel):
    """Result of POST /ingest across all uploaded files."""

    files: list[FileIngestResult]


class AskRequest(BaseModel):
    """A question, with an optional per-request top-k override."""

    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)


class Source(BaseModel):
    """One retrieved chunk surfaced as a citation in the answer."""

    source: str
    page: int
    chunk_index: int
    score: float
    content: str


class AskResponse(BaseModel):
    """A grounded answer plus the sources that supported it."""

    question: str
    answer: str
    sources: list[Source]
