"""Central configuration, loaded once from environment variables.

Every other module imports the singleton ``settings`` rather than reading
``os.environ`` directly, so provider choices and tuning knobs live in one place.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed view over the environment. See ``.env.example`` for the full list."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/ragdb"

    # Embeddings
    embedding_provider: Literal["local", "openai"] = "local"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # LLM
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Retrieval / chunking
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    backend_url: str = "http://localhost:8000"

    @property
    def embedding_dim(self) -> int:
        """Vector dimensionality for the active embedding provider.

        The pgvector column is fixed-width, so the schema must match whichever
        provider is configured at ingest time.
        """
        return 384 if self.embedding_provider == "local" else 1536


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached after first load)."""
    return Settings()


settings = get_settings()
