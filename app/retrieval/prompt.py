"""Prompt assembly for grounded answering.

The system prompt is the anti-hallucination contract: answer only from the
supplied context, cite the source/page behind each claim, and admit ignorance
rather than guess. Each context block is labelled ``[file p.N]`` so the model
has a concrete citation token to echo.
"""

from app.retrieval.retriever import RetrievedChunk

IDK = "I don't know based on the provided documents."

SYSTEM_PROMPT = f"""You are a precise assistant for querying industrial documents \
(equipment manuals, datasheets, maintenance guides).

Rules:
- Answer ONLY using the provided context. Do not use any outside knowledge.
- Each context block is labelled with its source as [file p.N]. After each claim, \
cite the label(s) that support it, e.g. "Torque the bolts to 40 Nm [pump.pdf p.3]".
- If the context does not contain enough information to answer, reply exactly: \
"{IDK}"
- Be concise and technical. Do not speculate or fill gaps."""


def _label(chunk: RetrievedChunk) -> str:
    """Citation token shown to the model and (later) to the user."""
    return f"[{chunk.source} p.{chunk.page}]"


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into a labelled context block for the prompt."""
    return "\n\n".join(f"{_label(c)}\n{c.content}" for c in chunks)


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Combine the labelled context and the question into the user turn."""
    return f"Context:\n{format_context(chunks)}\n\nQuestion: {question}"
