"""
evaluator.py
Part 2 – Step 7a: Model Evaluation (QAEvalChain pattern).

Implements a QA evaluation pipeline that:
  1. Defines a test suite of question / reference-answer pairs
  2. Generates model answers via the LLM engine
  3. Uses the LLM-as-judge pattern (mirrors LangChain's QAEvalChain) to score
     correctness, relevance, and groundedness
  4. Returns a structured results DataFrame
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd
import requests

API_URL = "https://api.anthropic.com/v1/messages"
MODEL   = "claude-3-5-sonnet-20241022"

# ── Test suite ─────────────────────────────────────────────────────────────────
QA_TEST_SET: list[dict[str, str]] = [
    {
        "question":  "What is the total revenue generated?",
        "reference": "The total revenue should be a large dollar figure in the millions derived from 2,000 orders.",
    },
    {
        "question":  "Which product generates the highest revenue?",
        "reference": "One of the Electronics products (Laptop Pro, Headphones Elite, or Monitor 27in) should be the top revenue generator.",
    },
    {
        "question":  "Which region performs best in terms of sales?",
        "reference": "One of the five regions (North, South, East, West, Central) leads in revenue.",
    },
    {
        "question":  "What is the profit margin percentage?",
        "reference": "The overall profit margin should be between 30% and 55% given the cost structure.",
    },
    {
        "question":  "Which customer segment contributes the most revenue?",
        "reference": "Enterprise or SMB segment typically contributes the most due to larger order sizes.",
    },
    {
        "question":  "What are the sales trends year over year?",
        "reference": "The data spans 2022–2024, so there should be three years of revenue figures showing growth or decline.",
    },
]


# ── Scoring rubric ─────────────────────────────────────────────────────────────
EVAL_SYSTEM = textwrap.dedent("""
    You are a strict but fair AI evaluation judge.
    You will be given:
      • A QUESTION
      • A REFERENCE answer (the gold standard)
      • A MODEL answer (to be evaluated)

    Score the model answer on three dimensions (each 1–5):
      1. Correctness   – Is the factual content accurate vs. the reference?
      2. Relevance     – Does the answer address what was asked?
      3. Groundedness  – Is the answer grounded in data rather than hallucinated?

    Respond ONLY with valid JSON in exactly this format (no markdown fences):
    {"correctness": <int>, "relevance": <int>, "groundedness": <int>, "feedback": "<one sentence>"}
""").strip()


@dataclass
class EvalResult:
    question:      str
    model_answer:  str
    correctness:   int
    relevance:     int
    groundedness:  int
    feedback:      str
    overall_score: float   # average of the three dimensions


# ══════════════════════════════════════════════════════════════════════════════
class QAEvaluator:
    """
    QAEvalChain-style evaluator.
    Usage:
        evaluator = QAEvaluator(llm_engine)
        results_df = evaluator.run()
    """

    def __init__(self, llm_engine: Any) -> None:
        self.engine = llm_engine

    # ── public API ─────────────────────────────────────────────────────────────
    def run(
        self,
        test_set: list[dict[str, str]] | None = None,
        progress_callback: Any = None,
    ) -> pd.DataFrame:
        """
        Runs evaluation over *test_set* and returns a DataFrame of EvalResult.
        *progress_callback* is an optional callable(current, total) for UI updates.
        """
        suite   = test_set or QA_TEST_SET
        results = []

        for idx, item in enumerate(suite, 1):
            if progress_callback:
                progress_callback(idx, len(suite))

            # generate model answer
            model_answer = self.engine.chat(item["question"])

            # judge it
            scores = self._judge(
                question     = item["question"],
                reference    = item["reference"],
                model_answer = model_answer,
            )

            results.append(
                EvalResult(
                    question      = item["question"],
                    model_answer  = model_answer[:300] + "…" if len(model_answer) > 300 else model_answer,
                    correctness   = scores["correctness"],
                    relevance     = scores["relevance"],
                    groundedness  = scores["groundedness"],
                    feedback      = scores["feedback"],
                    overall_score = round(
                        (scores["correctness"] + scores["relevance"] + scores["groundedness"]) / 3, 2
                    ),
                )
            )
            # clear memory between eval questions to avoid bleed-over
            self.engine.memory.clear()

        return pd.DataFrame([asdict(r) for r in results])

    def summary(self, results_df: pd.DataFrame) -> dict[str, float]:
        return {
            "avg_correctness":   round(results_df["correctness"].mean(),   2),
            "avg_relevance":     round(results_df["relevance"].mean(),     2),
            "avg_groundedness":  round(results_df["groundedness"].mean(),  2),
            "avg_overall":       round(results_df["overall_score"].mean(), 2),
            "total_questions":   len(results_df),
        }

    # ── internal judge call ────────────────────────────────────────────────────
    @staticmethod
    def _judge(question: str, reference: str, model_answer: str) -> dict[str, Any]:
        user_msg = textwrap.dedent(f"""
            QUESTION:
            {question}

            REFERENCE ANSWER:
            {reference}

            MODEL ANSWER:
            {model_answer}
        """).strip()

        payload = {
            "model":      MODEL,
            "max_tokens": 256,
            "system":     EVAL_SYSTEM,
            "messages":   [{"role": "user", "content": user_msg}],
        }
        resp = requests.post(API_URL, json=payload, headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        raw = resp.json()["content"][0]["text"].strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # fallback – extract JSON from possible markdown fences
            import re
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                return json.loads(m.group())
            return {"correctness": 3, "relevance": 3, "groundedness": 3,
                    "feedback": "Could not parse judge response."}
