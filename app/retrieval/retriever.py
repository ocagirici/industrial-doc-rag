"""Retrieval: a question in, the top-k most similar chunks out.

Cosine similarity is computed in Postgres via pgvector's ``<=>`` operator
(exposed as ``Column.cosine_distance``). Because we store unit-normalized
vectors, cosine distance ``d`` maps to similarity ``1 - d``.
"""

from dataclasses import dataclass

from sqlalchemy import select

from app.core.config import settings
from app.core.db import Chunk, SessionLocal
from app.ingestion.embedder import get_embedder


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by search, with its similarity to the query."""

    source: str
    page: int
    chunk_index: int
    content: str
    score: float


def retrieve(query: str, k: int | None = None) -> list[RetrievedChunk]:
    """Return the ``k`` chunks most similar to ``query``, best first.

    ``k`` defaults to ``settings.top_k``. The ANN ordering happens in the
    database; we only embed the query client-side.
    """
    k = k if k is not None else settings.top_k
    query_vector = get_embedder().embed_query(query)
    distance = Chunk.embedding.cosine_distance(query_vector)

    with SessionLocal() as session:
        rows = session.execute(
            select(Chunk, distance.label("distance")).order_by(distance).limit(k)
        ).all()

    return [
        RetrievedChunk(
            source=chunk.source,
            page=chunk.page,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            score=1.0 - distance,
        )
        for chunk, distance in rows
    ]
