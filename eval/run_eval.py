"""Evaluation harness: retrieval hit-rate + answer-quality over an eval set.

For each case we run the full RAG loop once and measure two things:
  * hit  — did the gold source (and page, if specified) appear in the retrieved
           top-k chunks? (retrieval quality)
  * keyword_score — fraction of expected keywords present in the answer, a cheap
           proxy for answer grounding (no LLM judge, no API cost beyond the answer)

Results are printed, saved to eval/results.csv, and written as a Markdown table
to eval/results.md for pasting into the README.

Run (after ingesting the referenced documents):
    python -m eval.run_eval
"""

import json
from pathlib import Path

import pandas as pd

from app.core.config import settings
from app.retrieval.answer import answer_question

EVAL_DIR = Path(__file__).parent
EVAL_SET = EVAL_DIR / "eval_set.json"
CSV_OUT = EVAL_DIR / "results.csv"
MD_OUT = EVAL_DIR / "results.md"


def keyword_score(answer: str, keywords: list[str]) -> float:
    """Fraction of expected keywords present in the answer (case-insensitive)."""
    if not keywords:
        return 1.0
    lowered = answer.lower()
    found = sum(1 for kw in keywords if kw.lower() in lowered)
    return found / len(keywords)


def evaluate_case(case: dict, top_k: int) -> dict:
    """Run one eval case through the RAG loop and score it."""
    result = answer_question(case["question"], k=top_k)

    retrieved = {(s.source, s.page) for s in result.sources}
    expected_source = case["expected_source"]
    expected_page = case.get("expected_page")
    if expected_page is None:
        hit = any(src == expected_source for src, _ in retrieved)
    else:
        hit = (expected_source, expected_page) in retrieved

    score = keyword_score(result.answer, case.get("keywords", []))

    return {
        "question": case["question"],
        "expected_source": expected_source,
        "hit": hit,
        "keyword_score": round(score, 2),
        "retrieved": ", ".join(sorted({src for src, _ in retrieved})) or "(none)",
        "answer": result.answer.replace("\n", " "),
    }


def to_markdown(df: pd.DataFrame) -> str:
    """Render the results as a Markdown table without extra dependencies."""
    cols = ["question", "expected_source", "hit", "keyword_score"]
    view = df[cols]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in view.values]
    return "\n".join([header, sep, *rows])


def main() -> None:
    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))["cases"]
    records = [evaluate_case(c, settings.top_k) for c in cases]
    df = pd.DataFrame(records)

    hit_rate = df["hit"].mean()
    mean_keyword = df["keyword_score"].mean()

    df.to_csv(CSV_OUT, index=False)
    summary = (
        f"**Eval summary** ({len(df)} cases, top_k={settings.top_k})\n\n"
        f"- Retrieval hit-rate: {hit_rate:.0%}\n"
        f"- Mean keyword score: {mean_keyword:.0%}\n"
    )
    MD_OUT.write_text(summary + "\n" + to_markdown(df) + "\n", encoding="utf-8")

    print(summary)
    print(to_markdown(df))
    print(f"\nwrote {CSV_OUT.name} and {MD_OUT.name}")


if __name__ == "__main__":
    main()
