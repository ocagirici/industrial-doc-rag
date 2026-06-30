"""Chunking: page-level text in, overlapping token-sized chunks out.

Sizes are measured in tokens (via tiktoken's ``cl100k_base``) rather than
characters, so ``CHUNK_SIZE``/``CHUNK_OVERLAP`` mean what the requirements say
they mean (~500-1000 tokens, 10-15% overlap). Each chunk keeps the page it came
from for citation, and gets a global index for stable ordering.
"""

from dataclasses import dataclass
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.ingestion.loader import Page


@dataclass(frozen=True)
class TextChunk:
    """A retrievable text span tied back to its source page."""

    source: str
    page: int
    chunk_index: int
    text: str


@lru_cache
def _splitter() -> RecursiveCharacterTextSplitter:
    """Build the token-aware recursive splitter once (encoder load is not free)."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def split_pages(pages: list[Page]) -> list[TextChunk]:
    """Split each page into overlapping chunks, numbered across the whole document.

    Overlap is applied within a page (we split page-by-page so a chunk never
    straddles a page boundary and muddies its citation).
    """
    chunks: list[TextChunk] = []
    for page in pages:
        for piece in _splitter().split_text(page.text):
            chunks.append(
                TextChunk(
                    source=page.source,
                    page=page.page,
                    chunk_index=len(chunks),
                    text=piece,
                )
            )
    return chunks
