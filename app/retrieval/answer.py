"""The RAG loop: question -> retrieve -> prompt -> generate -> grounded answer.

This is the readable core the whole project is built around — no framework
magic. Run directly to ask a question from the command line once chunks are
ingested and an API key is set:

    python -m app.retrieval.answer "What is the maximum operating pressure?"
"""

from dataclasses import dataclass

from app.retrieval.generator import get_generator
from app.retrieval.prompt import IDK, SYSTEM_PROMPT, build_user_prompt
from app.retrieval.retriever import RetrievedChunk, retrieve


@dataclass(frozen=True)
class Answer:
    """A generated answer plus the chunks that grounded it."""

    question: str
    answer: str
    sources: list[RetrievedChunk]


def answer_question(question: str, k: int | None = None) -> Answer:
    """Retrieve context for ``question`` and generate a grounded answer.

    If nothing is retrieved (empty store), we return the "I don't know" answer
    directly rather than prompting the model with no context.
    """
    chunks = retrieve(question, k)
    if not chunks:
        return Answer(question=question, answer=IDK, sources=[])

    user_prompt = build_user_prompt(question, chunks)
    text = get_generator().generate(SYSTEM_PROMPT, user_prompt)
    return Answer(question=question, answer=text.strip(), sources=chunks)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print('usage: python -m app.retrieval.answer "<question>"')
        raise SystemExit(1)

    result = answer_question(sys.argv[1])
    print(result.answer)
    print("\nsources:")
    for chunk in result.sources:
        print(f"  [{chunk.source} p.{chunk.page}] (score {chunk.score:.3f})")
