"""
visualizations.py
Part 2 – Step 7b: Data Visualization module.

All chart functions accept an optional `ax` (matplotlib Axes) argument so
they can be used both standalone and embedded in a multi-panel figure.
They also return the figure object so Streamlit can call st.pyplot(fig).

Charts produced:
  1. Sales trends over time (monthly line chart)
  2. Product performance comparison (horizontal bar)
  3. Regional analysis (choropleth-style bar)
  4. Customer demographics – gender pie + age-group bar
  5. Segment revenue comparison
  6. Profit margin by category
  7. Channel performance
  8. Revenue distribution (histogram + KDE)
"""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── colour palette ─────────────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#1a73e8",
    "secondary": "#34a853",
    "accent":    "#ea4335",
    "warn":      "#fbbc04",
    "purple":    "#7c3aed",
    "teal":      "#0f9d98",
    "bg":        "#f8fafc",
    "text":      "#1e293b",
}

REGION_COLORS = ["#1a73e8", "#34a853", "#ea4335", "#fbbc04", "#7c3aed"]
CAT_COLORS    = ["#1a73e8", "#34a853", "#ea4335"]

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.facecolor":     PALETTE["bg"],
    "figure.facecolor":   "white",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
})


# ══════════════════════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", color=PALETTE["text"], pad=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10, color=PALETTE["text"])
    if ylabel: ax.set_ylabel(ylabel, fontsize=10, color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"])


# ══════════════════════════════════════════════════════════════════════════════
# 1. Sales trend over time
# ══════════════════════════════════════════════════════════════════════════════

def plot_sales_trend(df: pd.DataFrame) -> plt.Figure:
    monthly = df.groupby("year_month")["revenue"].sum().reset_index()
    monthly["year_month"] = pd.PeriodIndex(monthly["year_month"], freq="M")
    monthly = monthly.sort_values("year_month")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(len(monthly)), monthly["revenue"] / 1_000,
            color=PALETTE["primary"], linewidth=2.2, marker="o", markersize=3)
    ax.fill_between(range(len(monthly)), monthly["revenue"] / 1_000,
                    alpha=0.12, color=PALETTE["primary"])

    # year separators
    current_year = None
    for i, p in enumerate(monthly["year_month"]):
        if p.year != current_year:
            ax.axvline(i, color="grey", linewidth=0.7, linestyle=":")
            ax.text(i + 0.3, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] else 1,
                    str(p.year), fontsize=8, color="grey")
            current_year = p.year

    # x-tick every 3 months
    ticks = list(range(0, len(monthly), 3))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(monthly["year_month"].iloc[t]) for t in ticks],
                       rotation=45, ha="right", fontsize=8)

    _fmt(ax, "Monthly Revenue Trend", ylabel="Revenue ($K)")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2. Product performance
# ══════════════════════════════════════════════════════════════════════════════

def plot_product_performance(df: pd.DataFrame) -> plt.Figure:
    prod = (df.groupby("product")[["revenue", "profit"]]
              .sum().sort_values("revenue", ascending=True))

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(prod))
    h = 0.35

    bars1 = ax.barh(y + h/2, prod["revenue"] / 1_000, h,
                    color=PALETTE["primary"], label="Revenue ($K)")
    bars2 = ax.barh(y - h/2, prod["profit"] / 1_000, h,
                    color=PALETTE["secondary"], label="Profit ($K)")

    ax.set_yticks(y)
    ax.set_yticklabels(prod.index, fontsize=9)
    ax.legend(fontsize=9)
    _fmt(ax, "Product Revenue vs Profit", xlabel="Amount ($K)")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. Regional analysis
# ══════════════════════════════════════════════════════════════════════════════

def plot_regional_analysis(df: pd.DataFrame) -> plt.Figure:
    reg = df.groupby("region")[["revenue", "profit"]].sum().sort_values("revenue", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # bar chart
    axes[0].bar(reg.index, reg["revenue"] / 1_000, color=REGION_COLORS, edgecolor="white")
    _fmt(axes[0], "Revenue by Region", ylabel="Revenue ($K)")

    # pie for profit share
    axes[1].pie(reg["profit"], labels=reg.index, colors=REGION_COLORS,
                autopct="%1.1f%%", startangle=140,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[1].set_title("Profit Share by Region", fontsize=13, fontweight="bold",
                       color=PALETTE["text"])

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4. Customer demographics
# ══════════════════════════════════════════════════════════════════════════════

def plot_demographics(df: pd.DataFrame) -> plt.Figure:
    gender = df.groupby("gender")["revenue"].sum()
    age    = df.groupby("age_group")["revenue"].sum().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # gender pie
    axes[0].pie(gender, labels=gender.index,
                colors=[PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]],
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[0].set_title("Revenue by Gender", fontsize=13, fontweight="bold",
                       color=PALETTE["text"])

    # age bar
    axes[1].bar(age.index, age / 1_000,
                color=plt.cm.Blues(np.linspace(0.4, 0.9, len(age))),  # type: ignore[arg-type]
                edgecolor="white")
    _fmt(axes[1], "Revenue by Age Group", xlabel="Age Group", ylabel="Revenue ($K)")

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. Customer segment comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_segment_analysis(df: pd.DataFrame) -> plt.Figure:
    seg = df.groupby("segment").agg(
        revenue    =("revenue", "sum"),
        avg_order  =("revenue", "mean"),
        order_count=("order_id","count"),
    ).sort_values("revenue", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"], PALETTE["purple"]]

    axes[0].bar(seg.index, seg["revenue"] / 1_000, color=colors, edgecolor="white")
    _fmt(axes[0], "Total Revenue by Segment", ylabel="Revenue ($K)")

    axes[1].bar(seg.index, seg["avg_order"], color=colors, edgecolor="white")
    _fmt(axes[1], "Avg Order Value by Segment", ylabel="Avg Order ($)")

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 6. Profit margin by category
# ══════════════════════════════════════════════════════════════════════════════

def plot_category_margin(df: pd.DataFrame) -> plt.Figure:
    cat = df.groupby("category").agg(revenue=("revenue","sum"), profit=("profit","sum"))
    cat["margin_pct"] = (cat["profit"] / cat["revenue"] * 100).round(1)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(cat.index, cat["margin_pct"], color=CAT_COLORS, edgecolor="white")
    for bar, val in zip(bars, cat["margin_pct"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

    ax.set_ylim(0, cat["margin_pct"].max() * 1.2)
    _fmt(ax, "Profit Margin % by Category", ylabel="Margin (%)")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 7. Channel performance
# ══════════════════════════════════════════════════════════════════════════════

def plot_channel_performance(df: pd.DataFrame) -> plt.Figure:
    ch = df.groupby("channel")[["revenue","profit"]].sum().sort_values("revenue", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(ch))
    w = 0.4
    ax.bar(x - w/2, ch["revenue"] / 1_000, w, color=PALETTE["primary"], label="Revenue ($K)")
    ax.bar(x + w/2, ch["profit"] / 1_000,  w, color=PALETTE["secondary"], label="Profit ($K)")
    ax.set_xticks(x)
    ax.set_xticklabels(ch.index)
    ax.legend()
    _fmt(ax, "Revenue & Profit by Sales Channel", ylabel="Amount ($K)")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 8. Revenue distribution
# ══════════════════════════════════════════════════════════════════════════════

def plot_revenue_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(df["revenue"], bins=40, color=PALETTE["teal"], edgecolor="white", alpha=0.8)
    ax.axvline(df["revenue"].mean(),   color=PALETTE["accent"], linewidth=2,
               linestyle="--", label=f"Mean  ${df['revenue'].mean():,.0f}")
    ax.axvline(df["revenue"].median(), color=PALETTE["warn"],   linewidth=2,
               linestyle=":",  label=f"Median ${df['revenue'].median():,.0f}")
    ax.legend()
    _fmt(ax, "Revenue Distribution per Order", xlabel="Order Revenue ($)", ylabel="Count")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard helper – all charts in one call
# ══════════════════════════════════════════════════════════════════════════════

CHART_REGISTRY: dict[str, Any] = {
    "Sales Trend":          plot_sales_trend,
    "Product Performance":  plot_product_performance,
    "Regional Analysis":    plot_regional_analysis,
    "Demographics":         plot_demographics,
    "Segment Analysis":     plot_segment_analysis,
    "Category Margin":      plot_category_margin,
    "Channel Performance":  plot_channel_performance,
    "Revenue Distribution": plot_revenue_distribution,
}


def render_chart(name: str, df: pd.DataFrame) -> plt.Figure:
    """Convenience wrapper: render chart by name."""
    fn = CHART_REGISTRY.get(name)
    if fn is None:
        raise ValueError(f"Unknown chart: {name!r}. Choose from {list(CHART_REGISTRY)}")
    return fn(df)
