"""
llm_engine.py
Part 1 – Steps 3, 4 & 6: LLM application, chain prompts, and memory integration.

Uses the Anthropic Messages API directly (no LangChain dependency needed)
while still following the RAG pattern described in the assignment:
  • Custom retriever  → fetches relevant stat blocks
  • Prompt engineering→ structured system + context + user question
  • Memory           → rolling conversation history (ConversationBufferMemory
                       pattern, implemented manually so the app works with or
                       without langchain installed)
"""

from __future__ import annotations

import textwrap
from typing import Any

import requests

# ── constants ─────────────────────────────────────────────────────────────────
API_URL = "https://api.anthropic.com/v1/messages"
MODEL   = "claude-sonnet-4-20250514"
MAX_TOK = 1024

# ══════════════════════════════════════════════════════════════════════════════
# PROMPT ENGINEERING  (Step 4 – chain prompts)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""
    You are InsightForge, an expert AI-powered Business Intelligence Assistant.
    Your sole purpose is to analyse the sales data provided in the CONTEXT block
    and answer the user's question with precise, data-driven insights.

    Guidelines:
    1. Always ground your answer in the numbers from CONTEXT — never hallucinate figures.
    2. Highlight key trends, comparisons, and anomalies the user should notice.
    3. End every answer with a short "💡 Recommendation" paragraph (1–3 sentences).
    4. Use $ for currency amounts and format large numbers with commas.
    5. If the CONTEXT does not contain enough data to answer, say so clearly and
       suggest what analysis would help.
    6. Keep the tone professional but approachable — avoid jargon overload.
""").strip()


def _build_user_message(context_blocks: list[str], question: str) -> str:
    """
    Constructs the user-turn message that injects retrieved context.
    This implements the RAG prompt pattern (Step 5).
    """
    context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "No specific data retrieved."
    return textwrap.dedent(f"""
        CONTEXT (retrieved from the business knowledge base):
        ────────────────────────────────────────────────────
        {context_text}
        ────────────────────────────────────────────────────

        USER QUESTION:
        {question}
    """).strip()


# ══════════════════════════════════════════════════════════════════════════════
# MEMORY  (Step 6 – conversation memory)
# ══════════════════════════════════════════════════════════════════════════════

class ConversationMemory:
    """
    Simple rolling-window conversation buffer.
    Mirrors the behaviour of LangChain's ConversationBufferWindowMemory.
    """

    def __init__(self, max_turns: int = 6) -> None:
        self.max_turns = max_turns          # keep last N user/assistant pairs
        self._history: list[dict[str, str]] = []

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        self._history.append({"role": "user",      "content": user_msg})
        self._history.append({"role": "assistant", "content": assistant_msg})
        # trim to window
        if len(self._history) > self.max_turns * 2:
            self._history = self._history[-(self.max_turns * 2):]

    def get_messages(self) -> list[dict[str, str]]:
        """Return the stored history in Anthropic messages format."""
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history) // 2          # number of full turns


# ══════════════════════════════════════════════════════════════════════════════
# LLM ENGINE  (Steps 3 & 4)
# ══════════════════════════════════════════════════════════════════════════════

class LLMEngine:
    """
    Wraps the Anthropic Messages API and wires together:
      • KnowledgeBase retriever  (RAG)
      • Prompt construction      (chain prompts)
      • ConversationMemory       (memory integration)
    """

    def __init__(self, retriever: Any, memory: ConversationMemory | None = None) -> None:
        self.retriever = retriever
        self.memory    = memory or ConversationMemory()

    # ── main entry point ──────────────────────────────────────────────────────
    def chat(self, question: str) -> str:
        """
        Full RAG pipeline:
          1. Retrieve relevant context chunks
          2. Build augmented prompt
          3. Prepend conversation history
          4. Call LLM
          5. Store turn in memory
        """
        # Step 1 – retrieve
        try:
            context_blocks = self.retriever.kb.retrieve(question, k=3)
        except AttributeError:
            context_blocks = []

        # Step 2 – build current user message (with injected context)
        user_msg = _build_user_message(context_blocks, question)

        # Step 3 – assemble full message list (history + current)
        messages = self.memory.get_messages() + [
            {"role": "user", "content": user_msg}
        ]

        # Step 4 – call LLM
        response_text = self._call_api(messages)

        # Step 5 – store in memory
        self.memory.add_turn(user_msg, response_text)

        return response_text

    # ── advanced data summary (Step 3 sub-task) ───────────────────────────────
    def generate_summary(self, kb_instance: Any) -> str:
        """
        Generates a comprehensive data summary covering:
          • Sales performance by time period
          • Product and regional analysis
          • Customer segmentation
          • Statistical measures
        """
        all_context = [chunk["text"] for chunk in kb_instance.chunks]
        prompt = textwrap.dedent("""
            Using ALL the context blocks provided, write a comprehensive
            Business Intelligence Summary Report with these sections:

            1. **Executive Summary** – 3 bullet key highlights
            2. **Sales Performance by Time Period** – year-over-year trends
            3. **Product & Category Analysis** – top performers, category breakdown
            4. **Regional Analysis** – best and worst regions with % difference
            5. **Customer Segmentation** – segment & demographic insights
            6. **Statistical Deep-Dive** – median, std deviation, outlier notes
            7. **Strategic Recommendations** – 3 actionable recommendations

            Be specific with numbers from the context. Format with clear headers.
        """).strip()

        user_msg = _build_user_message(all_context, prompt)
        return self._call_api([{"role": "user", "content": user_msg}])

    # ── internal API call ─────────────────────────────────────────────────────
    @staticmethod
    def _call_api(messages: list[dict]) -> str:
        # Read API key from Streamlit secrets (for deployment) or environment
        try:
            import streamlit as st
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            api_key = ""

        if not api_key:
            import os
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")

        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model":      MODEL,
            "max_tokens": MAX_TOK,
            "system":     SYSTEM_PROMPT,
            "messages":   messages,
        }
        resp = requests.post(API_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]
