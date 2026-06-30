"""Scoring engine for the eval harness — pure functions, no I/O.

Modelled on the constraint-based approach in ocagirici/agent-eval-harness: a case
asserts what the answer must / may / must-not contain, rather than a single fuzzy
keyword score. That's what lets the eval actually *fail* and discriminate.

Constraint fields on a case (all optional, combine freely):
  expected         — every term must appear            (AND)
  expected_any     — at least one term must appear     (OR)
  expected_exactly — these must appear AND none of `candidate_pool` besides them
  expected_none    — no term may appear  (catches over-answering / hallucination)
  must_refuse      — answer must be the "I don't know" refusal (adversarial / unanswerable)
  expected_page    — retrieval diagnostic: gold page should be in the retrieved top-k

On a must_refuse case, reserve expected_none for *fabricated specifics* (invented
numbers, codes, steps) — never the question's topic noun, since a correct refusal
restates the topic ("no information about Bluetooth") and would falsely fail.

Matching:
  * purely-numeric terms match on digit boundaries, so "5" matches "5V" but not
    "0.85" or "120", and thousands separators are normalized ("42,192" == "42192")
  * text terms match on word boundaries, so "up" doesn't match "support"
"""

import re
from dataclasses import dataclass

REFUSAL_MARKER = "i don't know"  # substring of the canonical IDK answer


@dataclass(frozen=True)
class CaseResult:
    """Outcome of scoring one case."""

    id: str
    difficulty: str
    question: str
    passed: bool
    category: str  # pass | retrieval_miss | hallucination | over_answer | missing_fact
    retrieval_hit: bool | None  # None when the case sets no expected_page
    detail: str
    answer: str


def _normalize(text: str) -> str:
    return text.lower().replace(",", "")


def _is_numeric(term: str) -> bool:
    return re.fullmatch(r"\d+(\.\d+)?", term.strip()) is not None


def contains(answer_norm: str, term: str) -> bool:
    """Is ``term`` present in the (already normalized) answer, boundary-aware?"""
    t = _normalize(term).strip()
    if not t:
        return True
    if _is_numeric(t):
        return re.search(rf"(?<![\d.]){re.escape(t)}(?![\d.])", answer_norm) is not None
    return re.search(rf"\b{re.escape(t)}\b", answer_norm) is not None


def score_case(case: dict, answer: str, retrieved_pages: set[int]) -> CaseResult:
    """Score one case's answer against its constraints and categorize any failure."""
    ans = _normalize(answer)

    refused = REFUSAL_MARKER in ans
    must_refuse = case.get("must_refuse", False)

    missing = [t for t in case.get("expected", []) if not contains(ans, t)]
    any_terms = case.get("expected_any", [])
    any_ok = (not any_terms) or any(contains(ans, t) for t in any_terms)
    forbidden = [t for t in case.get("expected_none", []) if contains(ans, t)]

    exactly = case.get("expected_exactly")
    exactly_missing: list[str] = []
    exactly_extra: list[str] = []
    if exactly is not None:
        pool = case.get("candidate_pool", exactly)
        exactly_missing = [t for t in exactly if not contains(ans, t)]
        exactly_extra = [t for t in pool if t not in exactly and contains(ans, t)]

    expected_page = case.get("expected_page")
    retrieval_hit = None if expected_page is None else (expected_page in retrieved_pages)

    common = dict(
        id=case["id"],
        difficulty=case.get("difficulty", "unspecified"),
        question=case["question"],
        retrieval_hit=retrieval_hit,
        answer=answer.replace("\n", " "),
    )

    # Adversarial: the answer should have refused.
    if must_refuse:
        if refused and not forbidden:
            return CaseResult(passed=True, category="pass", detail="refused as expected", **common)
        return CaseResult(
            passed=False,
            category="hallucination",
            detail="answered an unanswerable question"
            + (f"; leaked {forbidden}" if forbidden else ""),
            **common,
        )

    # Answerable cases — attribute the first failing constraint.
    if forbidden:
        return CaseResult(passed=False, category="over_answer", detail=f"forbidden present: {forbidden}", **common)
    if exactly_extra:
        return CaseResult(passed=False, category="over_answer", detail=f"unexpected extras: {exactly_extra}", **common)
    if missing or not any_ok or exactly_missing:
        gaps = missing + exactly_missing + ([f"any{any_terms}"] if not any_ok else [])
        category = "retrieval_miss" if retrieval_hit is False else "missing_fact"
        return CaseResult(passed=False, category=category, detail=f"missing: {gaps}", **common)

    return CaseResult(passed=True, category="pass", detail="all constraints satisfied", **common)
