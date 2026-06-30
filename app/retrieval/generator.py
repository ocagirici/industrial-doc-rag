"""Answer generation behind a provider-agnostic interface.

Two providers, selected by ``LLM_PROVIDER``:
  * ``anthropic`` — Claude (default: Haiku 4.5), via the Messages API
  * ``openai``    — GPT (default: gpt-4o-mini), via Chat Completions

This is the only module that knows which LLM backend is active. ``max_tokens``
is deliberately modest — grounded answers over retrieved context are short.
"""

from functools import lru_cache
from typing import Protocol

from app.core.config import settings

MAX_TOKENS = 1024


class Generator(Protocol):
    """Minimal generation interface shared by both providers."""

    def generate(self, system: str, user: str) -> str: ...


class AnthropicGenerator:
    """Claude Messages API backend."""

    def __init__(self, model: str, api_key: str) -> None:
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key)
        self._model = model

    def generate(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAIGenerator:
    """OpenAI Chat Completions backend."""

    def __init__(self, model: str, api_key: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


@lru_cache
def get_generator() -> Generator:
    """Return the configured generator singleton (client built once)."""
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is unset")
        return OpenAIGenerator(settings.openai_model, settings.openai_api_key)
    if not settings.anthropic_api_key:
        raise RuntimeError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is unset")
    return AnthropicGenerator(settings.anthropic_model, settings.anthropic_api_key)
