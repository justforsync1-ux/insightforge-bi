"""
knowledge_base.py
Part 1 – Steps 2 & 5: Knowledge-base creation and RAG system setup.

Loads the CSV, computes a rich statistics dictionary, and exposes a
custom LangChain-compatible retriever that returns relevant stat blocks
based on keyword matching — no vector store required, so it runs
entirely offline / without extra API costs.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Optional LangChain imports (graceful fallback) ─────────────────────────────
try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseRetriever = object          # type: ignore[misc,assignment]
    Document = dict                 # type: ignore[assignment,misc]


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & KNOWLEDGE-BASE CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

class KnowledgeBase:
    """
    Loads *sales_data.csv* and pre-computes an exhaustive statistics
    dictionary.  Each entry in ``self.chunks`` is a labelled text block
    that can be retrieved by keyword similarity.
    """

    def __init__(self, csv_path: str | Path = "data/sales_data.csv") -> None:
        self.df = self._load(csv_path)
        self.stats: dict[str, Any] = {}
        self.chunks: list[dict[str, str]] = []
        self._build()

    # ── loading ───────────────────────────────────────────────────────────────
    @staticmethod
    def _load(path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=["order_date"])
        df["year"]    = df["order_date"].dt.year
        df["month"]   = df["order_date"].dt.month
        df["quarter"] = df["order_date"].dt.quarter
        df["year_month"] = df["order_date"].dt.to_period("M").astype(str)
        return df

    # ── builder ───────────────────────────────────────────────────────────────
    def _build(self) -> None:
        df = self.df
        s: dict[str, Any] = {}

        # ── overall summary ───────────────────────────────────────────────────
        s["overview"] = {
            "total_orders":   int(len(df)),
            "total_revenue":  round(float(df["revenue"].sum()), 2),
            "total_profit":   round(float(df["profit"].sum()),  2),
            "avg_order_value":round(float(df["revenue"].mean()), 2),
            "median_revenue": round(float(df["revenue"].median()), 2),
            "std_revenue":    round(float(df["revenue"].std()),  2),
            "date_range":     f"{df['order_date'].min().date()} → {df['order_date'].max().date()}",
        }

        # ── sales by year ─────────────────────────────────────────────────────
        _yr = df.groupby("year")[["revenue", "profit"]].agg(["sum", "mean"]).round(2)
        s["sales_by_year"] = {
            "revenue": {
                "sum":  _yr[("revenue","sum")].to_dict(),
                "mean": _yr[("revenue","mean")].to_dict(),
            },
            "profit": {
                "sum":  _yr[("profit","sum")].to_dict(),
                "mean": _yr[("profit","mean")].to_dict(),
            },
        }

        # ── sales by quarter ──────────────────────────────────────────────────
        s["sales_by_quarter"] = (
            df.groupby(["year", "quarter"])["revenue"]
              .sum().round(2).to_dict()
        )

        # ── monthly trend ─────────────────────────────────────────────────────
        monthly = df.groupby("year_month")["revenue"].sum().round(2)
        s["monthly_trend"] = monthly.to_dict()

        # ── product analysis ──────────────────────────────────────────────────
        prod = df.groupby("product").agg(
            total_revenue =("revenue", "sum"),
            total_profit  =("profit",  "sum"),
            total_orders  =("order_id","count"),
            avg_unit_price=("unit_price","mean"),
        ).round(2).sort_values("total_revenue", ascending=False)
        s["product_analysis"] = prod.to_dict(orient="index")

        # ── category analysis ─────────────────────────────────────────────────
        cat = df.groupby("category").agg(
            total_revenue=("revenue","sum"),
            total_profit =("profit", "sum"),
            order_count  =("order_id","count"),
        ).round(2)
        s["category_analysis"] = cat.to_dict(orient="index")

        # ── regional analysis ─────────────────────────────────────────────────
        reg = df.groupby("region").agg(
            total_revenue=("revenue","sum"),
            total_profit =("profit", "sum"),
            order_count  =("order_id","count"),
        ).round(2).sort_values("total_revenue", ascending=False)
        s["regional_analysis"] = reg.to_dict(orient="index")

        # ── customer segmentation ─────────────────────────────────────────────
        seg = df.groupby("segment").agg(
            total_revenue=("revenue","sum"),
            order_count  =("order_id","count"),
            avg_order    =("revenue","mean"),
        ).round(2)
        s["segment_analysis"] = seg.to_dict(orient="index")

        # ── demographics ──────────────────────────────────────────────────────
        s["gender_analysis"] = (
            df.groupby("gender")["revenue"].agg(["sum","count","mean"])
              .round(2).to_dict(orient="index")
        )
        s["age_group_analysis"] = (
            df.groupby("age_group")["revenue"].agg(["sum","count","mean"])
              .round(2).to_dict(orient="index")
        )

        # ── channel analysis ──────────────────────────────────────────────────
        s["channel_analysis"] = (
            df.groupby("channel")["revenue"].agg(["sum","count","mean"])
              .round(2).to_dict(orient="index")
        )

        # ── top / bottom performers ───────────────────────────────────────────
        top5  = prod.head(5).index.tolist()
        bot5  = prod.tail(5).index.tolist()
        s["top_products"]    = top5
        s["bottom_products"] = bot5

        # ── statistical measures ──────────────────────────────────────────────
        s["statistics"] = {
            "revenue": {
                k: round(float(v), 2)
                for k, v in df["revenue"].describe().items()
            },
            "profit": {
                k: round(float(v), 2)
                for k, v in df["profit"].describe().items()
            },
            "profit_margin_pct": round(
                float(df["profit"].sum() / df["revenue"].sum() * 100), 2
            ),
        }

        self.stats = s
        self._make_chunks()

    # ── chunk builder ─────────────────────────────────────────────────────────
    def _make_chunks(self) -> None:
        s = self.stats
        ov = s["overview"]

        chunks = [
            {
                "label": "overview summary",
                "keywords": ["overview","summary","total","overall","general"],
                "text": textwrap.dedent(f"""
                    BUSINESS OVERVIEW
                    Orders      : {ov['total_orders']:,}
                    Revenue     : ${ov['total_revenue']:,.2f}
                    Profit      : ${ov['total_profit']:,.2f}
                    Avg Order   : ${ov['avg_order_value']:,.2f}
                    Median Rev  : ${ov['median_revenue']:,.2f}
                    Std Dev Rev : ${ov['std_revenue']:,.2f}
                    Period      : {ov['date_range']}
                """).strip(),
            },
            {
                "label": "sales time trend",
                "keywords": ["time","trend","year","quarter","monthly","period","growth"],
                "text": "YEARLY REVENUE:\n" + "\n".join(
                    f"  {yr}: ${rev:,.2f}"
                    for yr, rev in
                    pd.Series(s["sales_by_year"]["revenue"]["sum"]).items()
                ),
            },
            {
                "label": "product performance",
                "keywords": ["product","item","sku","performance","best","top","worst"],
                "text": "PRODUCT REVENUE (TOP 5):\n" + "\n".join(
                    f"  {p}: ${d['total_revenue']:,.2f}  profit ${d['total_profit']:,.2f}"
                    for p, d in list(s["product_analysis"].items())[:5]
                ),
            },
            {
                "label": "category analysis",
                "keywords": ["category","electronics","furniture","stationery"],
                "text": "CATEGORY ANALYSIS:\n" + "\n".join(
                    f"  {c}: revenue ${d['total_revenue']:,.2f}, orders {d['order_count']}"
                    for c, d in s["category_analysis"].items()
                ),
            },
            {
                "label": "regional analysis",
                "keywords": ["region","north","south","east","west","central","geography","location"],
                "text": "REGIONAL REVENUE:\n" + "\n".join(
                    f"  {r}: ${d['total_revenue']:,.2f}  ({d['order_count']} orders)"
                    for r, d in s["regional_analysis"].items()
                ),
            },
            {
                "label": "customer segmentation",
                "keywords": ["segment","enterprise","smb","startup","individual","customer","demographic"],
                "text": "CUSTOMER SEGMENTS:\n" + "\n".join(
                    f"  {seg}: revenue ${d['total_revenue']:,.2f}, avg order ${d['avg_order']:,.2f}"
                    for seg, d in s["segment_analysis"].items()
                ),
            },
            {
                "label": "demographics gender age",
                "keywords": ["gender","male","female","age","demographic","group"],
                "text": (
                    "GENDER BREAKDOWN:\n" + "\n".join(
                        f"  {g}: ${d['sum']:,.2f} ({d['count']} orders)"
                        for g, d in s["gender_analysis"].items()
                    ) + "\n\nAGE GROUP BREAKDOWN:\n" + "\n".join(
                        f"  {a}: ${d['sum']:,.2f} ({d['count']} orders)"
                        for a, d in s["age_group_analysis"].items()
                    )
                ),
            },
            {
                "label": "channel analysis",
                "keywords": ["channel","online","retail","direct","partner","sales channel"],
                "text": "CHANNEL PERFORMANCE:\n" + "\n".join(
                    f"  {ch}: ${d['sum']:,.2f}  avg ${d['mean']:,.2f}"
                    for ch, d in s["channel_analysis"].items()
                ),
            },
            {
                "label": "statistics measures",
                "keywords": ["statistic","median","std","standard deviation","mean","min","max","distribution"],
                "text": textwrap.dedent(f"""
                    STATISTICAL MEASURES
                    Revenue  – mean ${s['statistics']['revenue']['mean']:,.2f}
                               median ${s['statistics']['revenue']['50%']:,.2f}
                               std ${s['statistics']['revenue']['std']:,.2f}
                               min ${s['statistics']['revenue']['min']:,.2f}
                               max ${s['statistics']['revenue']['max']:,.2f}
                    Profit   – mean ${s['statistics']['profit']['mean']:,.2f}
                               std ${s['statistics']['profit']['std']:,.2f}
                    Margin   – {s['statistics']['profit_margin_pct']}%
                """).strip(),
            },
        ]
        self.chunks = chunks

    # ── public helpers ────────────────────────────────────────────────────────
    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Return the *k* most relevant chunk texts for *query*."""
        q_words = set(re.sub(r"[^\w\s]", "", query.lower()).split())
        scored  = []
        for chunk in self.chunks:
            kw_hits = sum(1 for kw in chunk["keywords"] if kw in query.lower())
            word_hits = sum(1 for w in q_words if w in chunk["text"].lower())
            scored.append((kw_hits + word_hits * 0.5, chunk["text"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:k]]

    def get_dataframe(self) -> pd.DataFrame:
        return self.df

    def get_stats_json(self) -> str:
        return json.dumps(self.stats, indent=2, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# 2. LANGCHAIN-COMPATIBLE CUSTOM RETRIEVER (Step 5)
# ══════════════════════════════════════════════════════════════════════════════

if LANGCHAIN_AVAILABLE:
    from pydantic import Field

    class InsightRetriever(BaseRetriever):
        """
        Custom LangChain retriever wrapping KnowledgeBase.retrieve().
        Compatible with LangChain's LCEL / chain interfaces.
        """
        kb: Any = Field(description="KnowledgeBase instance")
        k: int  = Field(default=3, description="Number of chunks to return")

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun | None = None,
        ) -> list[Document]:
            texts = self.kb.retrieve(query, k=self.k)
            return [Document(page_content=t) for t in texts]

else:
    class InsightRetriever:  # type: ignore[no-redef]
        """Fallback when LangChain is not installed."""
        def __init__(self, kb: KnowledgeBase, k: int = 3) -> None:
            self.kb = kb
            self.k  = k

        def get_relevant_documents(self, query: str) -> list[dict]:
            return [{"page_content": t} for t in self.kb.retrieve(query, self.k)]
