"""Embeddings: text in, vectors out, behind a provider-agnostic interface.

Two providers, selected by ``EMBEDDING_PROVIDER``:
  * ``local``  — sentence-transformers ``all-MiniLM-L6-v2`` (384-dim, no API cost)
  * ``openai`` — ``text-embedding-3-small`` (1536-dim)

Vectors are L2-normalized so a cosine-distance ANN search behaves like cosine
similarity. This is the only module that knows which embedding backend is active.
"""

from functools import lru_cache
from typing import Protocol

from app.core.config import settings


class Embedder(Protocol):
    """Minimal embedding interface shared by both providers."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LocalEmbedder:
    """sentence-transformers backend; runs on CPU, no network at query time."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OpenAIEmbedder:
    """OpenAI embeddings backend; normalizes client-side for cosine parity."""

    def __init__(self, model_name: str, api_key: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [_normalize(item.embedding) for item in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _normalize(vec: list[float]) -> list[float]:
    """Scale a vector to unit length (no-op if it is already ~zero)."""
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0:
        return vec
    return [x / norm for x in vec]


@lru_cache
def get_embedder() -> Embedder:
    """Return the configured embedder singleton (model load happens once)."""
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is unset")
        return OpenAIEmbedder(settings.openai_embedding_model, settings.openai_api_key)
    return LocalEmbedder(settings.local_embedding_model)
