"""Database engine, the ``chunks`` table, and schema bootstrap.

The vector column width is fixed to the *active* embedding provider's
dimensionality (``settings.embedding_dim``). Switching providers therefore means
re-ingesting, since a 384-dim MiniLM vector cannot live in a 1536-dim column.
"""

from sqlalchemy import Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from pgvector.sqlalchemy import Vector

from app.core.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


class Base(DeclarativeBase):
    pass


class Chunk(Base):
    """One retrievable unit: a slice of a source PDF plus its embedding."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(512), index=True)
    page: Mapped[int] = mapped_column(Integer)
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dim))


def init_db() -> None:
    """Create the pgvector extension, the ``chunks`` table, and its ANN index.

    Idempotent — safe to call on every app startup. The HNSW index uses cosine
    distance (``vector_cosine_ops``), matching the normalized embeddings we store.
    """
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(bind=conn)
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
                "ON chunks USING hnsw (embedding vector_cosine_ops)"
            )
        )
