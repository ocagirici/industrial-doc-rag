"""Evaluation runner: scores a testset against the live RAG loop.

Each case runs through ``answer_question`` once; the answer is scored against the
case's constraints by ``evaluator.score_case`` (see that module for the schema).
Reports overall pass-rate, retrieval hit-rate, a per-difficulty breakdown, and a
failure-category breakdown — then writes results.csv and results.md.

Run (after ingesting the referenced documents):
    python -m eval.run_eval
    python -m eval.run_eval --testset eval/my_cases.json --top-k 8
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from app.core.config import settings
from app.retrieval.answer import answer_question
from eval.evaluator import CaseResult, score_case

EVAL_DIR = Path(__file__).parent
CSV_OUT = EVAL_DIR / "results.csv"
MD_OUT = EVAL_DIR / "results.md"


def run_case(case: dict, top_k: int) -> CaseResult:
    """Execute the RAG loop for one case and score the answer."""
    result = answer_question(case["question"], k=top_k)
    retrieved_pages = {s.page for s in result.sources}
    return score_case(case, result.answer, retrieved_pages)


def build_report(results: list[CaseResult], top_k: int) -> tuple[pd.DataFrame, str]:
    """Build the results DataFrame and a Markdown report."""
    df = pd.DataFrame([r.__dict__ for r in results])

    passed = int(df["passed"].sum())
    total = len(df)
    scored_retrieval = df[df["retrieval_hit"].notna()]
    hit_rate = scored_retrieval["retrieval_hit"].mean() if len(scored_retrieval) else float("nan")

    by_diff = (
        df.groupby("difficulty")["passed"].agg(["sum", "count"]).reset_index()
    )
    diff_lines = [
        f"- {row.difficulty}: {int(row['sum'])}/{int(row['count'])}"
        for _, row in by_diff.iterrows()
    ]

    fails = df[~df["passed"]]
    cat_lines = [
        f"- {cat}: {n}" for cat, n in fails["category"].value_counts().items()
    ] or ["- (none)"]

    case_cols = ["id", "difficulty", "retrieval_hit", "passed", "category", "detail"]
    header = "| " + " | ".join(case_cols) + " |"
    sep = "| " + " | ".join("---" for _ in case_cols) + " |"
    rows = [
        "| " + " | ".join(str(df.loc[i, c]) for c in case_cols) + " |"
        for i in df.index
    ]

    hit_str = "n/a" if scored_retrieval.empty else f"{hit_rate:.0%}"
    report = "\n".join(
        [
            f"## Eval results ({total} cases, top_k={top_k})",
            "",
            f"- **Answer pass-rate: {passed}/{total} ({passed / total:.0%})**",
            f"- Retrieval hit-rate (page-level, {len(scored_retrieval)} answerable cases): {hit_str}",
            "",
            "**By difficulty:**",
            *diff_lines,
            "",
            "**Failures by category:**",
            *cat_lines,
            "",
            header,
            sep,
            *rows,
            "",
        ]
    )
    return df, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG eval harness.")
    parser.add_argument("--testset", default=str(EVAL_DIR / "testset.json"))
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    args = parser.parse_args()

    cases = json.loads(Path(args.testset).read_text(encoding="utf-8"))["cases"]
    results = [run_case(c, args.top_k) for c in cases]

    df, report = build_report(results, args.top_k)
    df.to_csv(CSV_OUT, index=False)
    MD_OUT.write_text(report, encoding="utf-8")

    print(report)
    print(f"wrote {CSV_OUT.name} and {MD_OUT.name}")


if __name__ == "__main__":
    main()
