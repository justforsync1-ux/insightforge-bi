"""
app.py
Part 2 – Step 7c: Streamlit UI for InsightForge BI Assistant.

Launch with:
    streamlit run app.py

Tabs:
  📊 Dashboard       – KPI cards + all visualisations
  💬 AI Chat         – RAG-powered conversational BI assistant
  📑 Data Summary    – Auto-generated LLM business report
  🔬 Evaluation      – QAEvalChain-style model evaluation
  📁 Raw Data        – Dataset explorer
"""

import sys
import os

# ── ensure project root is importable ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from components.knowledge_base import KnowledgeBase, InsightRetriever
from components.llm_engine      import LLMEngine, ConversationMemory
from utils.visualizations       import CHART_REGISTRY, render_chart
from evaluations.evaluator      import QAEvaluator

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="InsightForge – AI BI Assistant",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background: #f0f4f8; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0f172a 100%);
        color: white;
    }
    section[data-testid="stSidebar"] * { color: white !important; }

    /* KPI cards */
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid;
    }
    .kpi-value { font-size: 1.7rem; font-weight: 700; margin: 4px 0; }
    .kpi-label { font-size: 0.85rem; color: #64748b; text-transform: uppercase;
                 letter-spacing: .05em; }

    /* Chat bubbles */
    .chat-user {
        background: #1a73e8; color: white; border-radius: 18px 18px 4px 18px;
        padding: 10px 16px; margin: 6px 0; max-width: 78%; margin-left: auto;
        font-size: 0.95rem;
    }
    .chat-bot {
        background: white; color: #1e293b; border-radius: 18px 18px 18px 4px;
        padding: 10px 16px; margin: 6px 0; max-width: 82%;
        border: 1px solid #e2e8f0; font-size: 0.95rem;
    }

    /* Section headers */
    h2 { color: #1e293b !important; }
    .stTab [data-baseweb="tab"] { font-size: 0.9rem; }

    /* Eval table */
    .eval-high  { color: #16a34a; font-weight: 600; }
    .eval-mid   { color: #d97706; font-weight: 600; }
    .eval-low   { color: #dc2626; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION-STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading knowledge base…")
def load_kb() -> KnowledgeBase:
    return KnowledgeBase("data/sales_data.csv")


def get_engine() -> LLMEngine:
    if "engine" not in st.session_state:
        kb       = load_kb()
        retriever = InsightRetriever(kb=kb)
        memory   = ConversationMemory(max_turns=6)
        st.session_state.engine = LLMEngine(retriever=retriever, memory=memory)
    return st.session_state.engine


if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []

if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

if "summary_text" not in st.session_state:
    st.session_state.summary_text = None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://via.placeholder.com/200x60/1a73e8/ffffff?text=InsightForge",
             use_container_width=True)
    st.markdown("### 🔮 AI BI Assistant")
    st.markdown("Powered by Claude + RAG")
    st.divider()

    kb = load_kb()
    ov = kb.stats["overview"]
    st.markdown("**Dataset at a glance**")
    st.markdown(f"- Orders : **{ov['total_orders']:,}**")
    st.markdown(f"- Revenue: **${ov['total_revenue']/1e6:.2f}M**")
    st.markdown(f"- Profit : **${ov['total_profit']/1e6:.2f}M**")
    st.markdown(f"- Period : **{ov['date_range']}**")
    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        get_engine().memory.clear()
        st.rerun()

    st.markdown("---")
    st.caption("Advanced Generative AI | Capstone Project")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_dash, tab_chat, tab_summary, tab_eval, tab_data = st.tabs([
    "📊 Dashboard", "💬 AI Chat", "📑 Data Summary", "🔬 Evaluation", "📁 Raw Data"
])

df = kb.get_dataframe()


# ── TAB 1: DASHBOARD ──────────────────────────────────────────────────────────
with tab_dash:
    st.markdown("## 📊 Business Intelligence Dashboard")

    # KPI row
    kpis = [
        ("Total Revenue",  f"${ov['total_revenue']/1e6:.2f}M", "#1a73e8"),
        ("Total Profit",   f"${ov['total_profit']/1e6:.2f}M",  "#34a853"),
        ("Profit Margin",  f"{ov['total_profit']/ov['total_revenue']*100:.1f}%", "#7c3aed"),
        ("Avg Order",      f"${ov['avg_order_value']:,.0f}",    "#ea4335"),
        ("Total Orders",   f"{ov['total_orders']:,}",           "#fbbc04"),
        ("Std Dev (Rev)",  f"${ov['std_revenue']:,.0f}",        "#0f9d98"),
    ]

    cols = st.columns(len(kpis))
    for col, (label, value, color) in zip(cols, kpis):
        col.markdown(f"""
            <div class="kpi-card" style="border-color:{color}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color}">{value}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Chart selector
    chart_names = list(CHART_REGISTRY.keys())
    selected = st.selectbox("Select Visualization", chart_names, key="chart_select")

    fig = render_chart(selected, df)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Show two at once option
    st.markdown("### Compare Two Charts")
    c1, c2 = st.columns(2)
    with c1:
        s1 = st.selectbox("Chart A", chart_names, index=0, key="c1")
        f1 = render_chart(s1, df)
        st.pyplot(f1, use_container_width=True)
        plt.close(f1)
    with c2:
        s2 = st.selectbox("Chart B", chart_names, index=2, key="c2")
        f2 = render_chart(s2, df)
        st.pyplot(f2, use_container_width=True)
        plt.close(f2)


# ── TAB 2: AI CHAT ────────────────────────────────────────────────────────────
with tab_chat:
    st.markdown("## 💬 AI-Powered BI Chat")
    st.caption("Ask questions about your sales data. The assistant uses RAG to ground answers in real data.")

    # Suggested questions
    st.markdown("**Quick questions:**")
    suggestions = [
        "What is the total revenue and profit?",
        "Which product performs best?",
        "Compare regions by sales",
        "Analyse customer demographics",
        "What are the sales trends year over year?",
    ]
    cols = st.columns(len(suggestions))
    for col, q in zip(cols, suggestions):
        if col.button(q, key=f"sugg_{q[:10]}"):
            st.session_state.pending_question = q

    st.divider()

    # Display history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🤖 {msg["content"]}</div>',
                        unsafe_allow_html=True)

    # Input
    user_input = st.chat_input("Ask about your business data…")

    # Handle pending suggestion click
    if "pending_question" in st.session_state:
        user_input = st.session_state.pop("pending_question")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("🔍 Retrieving data & generating insight…"):
            try:
                answer = get_engine().chat(user_input)
            except Exception as e:
                answer = f"⚠️ Error contacting AI: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()


# ── TAB 3: DATA SUMMARY ───────────────────────────────────────────────────────
with tab_summary:
    st.markdown("## 📑 AI-Generated Business Report")
    st.caption("LLM analyses all knowledge-base chunks to produce a comprehensive BI report.")

    if st.button("🚀 Generate Full Report", type="primary", use_container_width=True):
        with st.spinner("Analysing all data and generating report (takes ~20 s)…"):
            try:
                report = get_engine().generate_summary(kb)
                st.session_state.summary_text = report
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.summary_text:
        st.markdown(st.session_state.summary_text)
        st.download_button(
            "📥 Download Report",
            data=st.session_state.summary_text,
            file_name="insightforge_report.md",
            mime="text/markdown",
        )


# ── TAB 4: EVALUATION ─────────────────────────────────────────────────────────
with tab_eval:
    st.markdown("## 🔬 Model Evaluation (QAEvalChain)")
    st.caption(
        "Runs a QA test suite: the model answers each question, then an LLM judge "
        "scores correctness, relevance, and groundedness (1–5 scale)."
    )

    if st.button("▶️ Run Evaluation", type="primary", use_container_width=True):
        progress = st.progress(0, text="Initialising…")
        results_placeholder = st.empty()

        def progress_cb(current: int, total: int) -> None:
            pct = int(current / total * 100)
            progress.progress(pct, text=f"Evaluating question {current}/{total}…")

        with st.spinner("Running evaluation suite…"):
            try:
                evaluator = QAEvaluator(get_engine())
                results_df = evaluator.run(progress_callback=progress_cb)
                st.session_state.eval_results = results_df
                get_engine().memory.clear()
            except Exception as e:
                st.error(f"Evaluation error: {e}")

        progress.empty()

    if st.session_state.eval_results is not None:
        r = st.session_state.eval_results

        # Summary metrics
        st.markdown("### Overall Scores")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Correctness",  f"{r['correctness'].mean():.2f} / 5")
        m2.metric("Avg Relevance",    f"{r['relevance'].mean():.2f} / 5")
        m3.metric("Avg Groundedness", f"{r['groundedness'].mean():.2f} / 5")
        m4.metric("Overall Score",    f"{r['overall_score'].mean():.2f} / 5")

        st.markdown("### Detailed Results")

        # Colour-coded table
        def colour_score(val: float) -> str:
            if val >= 4:   return "background-color: #dcfce7"
            if val >= 3:   return "background-color: #fef9c3"
            return "background-color: #fee2e2"

        display_cols = ["question", "correctness", "relevance", "groundedness",
                        "overall_score", "feedback"]
        styled = (
            r[display_cols]
              .style
              .applymap(colour_score, subset=["correctness", "relevance",
                                              "groundedness", "overall_score"])
              .format({"correctness": "{:.0f}", "relevance": "{:.0f}",
                       "groundedness": "{:.0f}", "overall_score": "{:.2f}"})
        )
        st.dataframe(styled, use_container_width=True, height=320)

        # Score radar chart
        avg_scores = {
            "Correctness":  r["correctness"].mean(),
            "Relevance":    r["relevance"].mean(),
            "Groundedness": r["groundedness"].mean(),
        }
        fig_ev, ax_ev = plt.subplots(figsize=(6, 3))
        bars_ev = ax_ev.barh(list(avg_scores.keys()), list(avg_scores.values()),
                             color=["#1a73e8", "#34a853", "#7c3aed"])
        ax_ev.set_xlim(0, 5)
        ax_ev.axvline(3, color="grey", linestyle="--", linewidth=0.8, label="Baseline (3)")
        for b, v in zip(bars_ev, avg_scores.values()):
            ax_ev.text(v + 0.05, b.get_y() + b.get_height()/2,
                       f"{v:.2f}", va="center", fontsize=10, fontweight="bold")
        ax_ev.set_title("Average Evaluation Scores", fontweight="bold")
        ax_ev.legend()
        st.pyplot(fig_ev, use_container_width=True)
        plt.close(fig_ev)

        st.download_button(
            "📥 Download Evaluation CSV",
            data=r.to_csv(index=False),
            file_name="eval_results.csv",
            mime="text/csv",
        )


# ── TAB 5: RAW DATA ───────────────────────────────────────────────────────────
with tab_data:
    st.markdown("## 📁 Dataset Explorer")

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sel_regions = st.multiselect("Filter by Region",
                                     options=df["region"].unique().tolist(),
                                     default=df["region"].unique().tolist())
    with fc2:
        sel_cats = st.multiselect("Filter by Category",
                                  options=df["category"].unique().tolist(),
                                  default=df["category"].unique().tolist())
    with fc3:
        sel_segs = st.multiselect("Filter by Segment",
                                  options=df["segment"].unique().tolist(),
                                  default=df["segment"].unique().tolist())

    filtered = df[
        df["region"].isin(sel_regions) &
        df["category"].isin(sel_cats) &
        df["segment"].isin(sel_segs)
    ]

    st.markdown(f"Showing **{len(filtered):,}** of **{len(df):,}** records")
    st.dataframe(
        filtered.style.format({"revenue": "${:,.2f}", "profit": "${:,.2f}",
                               "unit_price": "${:,.2f}", "discount": "{:.0%}"}),
        use_container_width=True,
        height=420,
    )

    st.download_button(
        "📥 Download Filtered CSV",
        data=filtered.to_csv(index=False),
        file_name="filtered_sales.csv",
        mime="text/csv",
    )

    # Basic describe
    st.markdown("### Descriptive Statistics")
    st.dataframe(
        filtered[["revenue","profit","quantity","unit_price"]].describe().round(2),
        use_container_width=True,
    )
