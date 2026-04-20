"""
llm_engine.py - LLM Engine with RAG, Prompt Engineering and Memory
"""
from __future__ import annotations
import textwrap
from typing import Any
import requests

API_URL = "https://api.anthropic.com/v1/messages"
MODEL   = "claude-3-haiku-20240307"
MAX_TOK = 1024

SYSTEM_PROMPT = textwrap.dedent("""
    You are InsightForge, an expert AI-powered Business Intelligence Assistant.
    Always ground your answer in the numbers from CONTEXT — never hallucinate figures.
    Highlight key trends, comparisons, and anomalies the user should notice.
    End every answer with a short Recommendation paragraph (1-3 sentences).
    Use $ for currency amounts and format large numbers with commas.
    If the CONTEXT does not contain enough data to answer, say so clearly.
""").strip()


def _build_user_message(context_blocks: list[str], question: str) -> str:
    context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "No specific data retrieved."
    return f"CONTEXT (retrieved from the business knowledge base):\n{context_text}\n\nUSER QUESTION:\n{question}"


class ConversationMemory:
    def __init__(self, max_turns: int = 6) -> None:
        self.max_turns = max_turns
        self._history: list[dict] = []

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        self._history.append({"role": "user",      "content": user_msg})
        self._history.append({"role": "assistant", "content": assistant_msg})
        if len(self._history) > self.max_turns * 2:
            self._history = self._history[-(self.max_turns * 2):]

    def get_messages(self) -> list[dict]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history) // 2


class LLMEngine:
    def __init__(self, retriever: Any, memory: ConversationMemory | None = None) -> None:
        self.retriever = retriever
        self.memory    = memory or ConversationMemory()

    def chat(self, question: str) -> str:
        try:
            context_blocks = self.retriever.kb.retrieve(question, k=3)
        except Exception:
            context_blocks = []

        user_msg = _build_user_message(context_blocks, question)
        messages = self.memory.get_messages() + [{"role": "user", "content": user_msg}]
        response = self._call_api(messages)
        self.memory.add_turn(user_msg, response)
        return response

    def generate_summary(self, kb_instance: Any) -> str:
        all_context = [chunk["text"] for chunk in kb_instance.chunks]
        prompt = "Write a comprehensive BI Summary Report covering: 1. Executive Summary 2. Sales by Time Period 3. Product & Category Analysis 4. Regional Analysis 5. Customer Segmentation 6. Statistical Deep-Dive 7. Strategic Recommendations. Be specific with numbers."
        user_msg = _build_user_message(all_context, prompt)
        return self._call_api([{"role": "user", "content": user_msg}])

    @staticmethod
    def _get_api_key() -> str:
        # Try Streamlit secrets first
        try:
            import streamlit as st
            key = st.secrets.get("ANTHROPIC_API_KEY", "")
            if key:
                return key
        except Exception:
            pass
        # Fall back to environment variable
        import os
        return os.environ.get("ANTHROPIC_API_KEY", "")

    @staticmethod
    def _call_api(messages: list[dict]) -> str:
        api_key = LLMEngine._get_api_key()
        if not api_key:
            return "⚠️ No API key found. Please add ANTHROPIC_API_KEY to Streamlit secrets."

        headers = {
            "content-type":      "application/json",
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model":      MODEL,
            "max_tokens": MAX_TOK,
            "system":     SYSTEM_PROMPT,
            "messages":   messages,
        }
        resp = requests.post(API_URL, json=payload, headers=headers)
        if not resp.ok:
            return f"⚠️ API Error {resp.status_code}: {resp.text[:300]}"
        return resp.json()["content"][0]["text"]
