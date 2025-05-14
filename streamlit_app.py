#!/usr/bin/env python3
"""
Streamlit dashboard to compare QQQ, QQQ3.MI and a synthetic 3×-QQQ proxy.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""
from __future__ import annotations

import logging
from datetime import date
from functools import lru_cache

import altair as alt
import pandas as pd
import streamlit as st
import yfinance as yf

# ──────────────────────────────────────────
# Streamlit & logging
# ──────────────────────────────────────────
st.set_page_config(page_title="QQQ 3× ETF Comparison", layout="wide")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

TICKER_BASE = "QQQ"        # Nasdaq-100 ETF
TICKER_LEV  = "QQQ3.MI"    # 3× leveraged Nasdaq-100 ETF (Borsa Italiana)

# ──────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────
@lru_cache(maxsize=8)
def _download(symbol: str, start: str) -> pd.DataFrame:
    """
    Return one tz-naive adjusted-price column named exactly *symbol*.
    """
    df = yf.download(symbol, start=start, auto_adjust=True, progress=False)

    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if col not in df.columns:
        raise ValueError(f"{symbol}: missing price column")

    if df.index.tz is not None:                # drop timezone → tz-naive
        df.index = df.index.tz_convert(None)

    return df[[col]].rename(columns={col: symbol}).dropna()


def build_dataset(start: str) -> pd.DataFrame:
    """
    QQQ, QQQ3.MI + synthetic «QQQ×3» merged on common dates.
    """
    qqq  = _download(TICKER_BASE, start)
    qqq3 = _download(TICKER_LEV,  start)

    merged = pd.concat([qqq, qqq3], axis=1).dropna(how="all")
    merged["QQQ×3"] = merged[TICKER_BASE] * 3        # proxy
    merged = merged.dropna()

    if merged.empty:
        raise ValueError("No overlapping data for the chosen start date.")
    return merged


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Re-base each column so the first value = 100."""
    return df if df.empty else df.div(df.iloc[0]).mul(100)

# ──────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────
def main() -> None:
    st.title("📈 Nasdaq-100 – 3× ETF Comparison")

    start_date = st.sidebar.date_input(
        "Start date",
        value=date(2020, 1, 1),
        min_value=date(2000, 1, 1),
        max_value=date.today(),
    )
    view = st.sidebar.radio("Display mode", ("Raw price", "Normalised (start = 100)"))

    # --- data -----------------------------------------------------------------
    try:
        data = build_dataset(start_date.isoformat())
    except ValueError as e:
        st.error(str(e))
        st.stop()

    df_plot = normalise(data) if "Normalised" in view else data
    y_label = "Indexed level (start = 100)" if "Normalised" in view else "Price"

    # --- Altair chart (index fixed) -------------------------------------------
    chart_df = (
        df_plot
        .rename_axis("Date")        # ★ guarantees a 'Date' column
        .reset_index()
        .melt(id_vars="Date", var_name="Ticker", value_name="Price")
    )

    chart = (
        alt.Chart(chart_df)
        .mark_line()
        .encode(
            x="Date:T",
            y=alt.Y("Price:Q", title=y_label),
            color="Ticker:N",
            tooltip=["Date:T", "Ticker:N", alt.Tooltip("Price:Q", format=".2f")],
        )
        .properties(height=450)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)

    # --- latest rows ----------------------------------------------------------
    st.subheader("Latest snapshot")
    st.dataframe(df_plot.tail(3).style.format("{:.2f}"), use_container_width=True)
    st.caption(
        "**QQQ×3** is a simple 3× multiple of QQQ (ignores daily compounding); "
        "**QQQ3.MI** is the actual 3× ETF listed on Borsa Italiana."
    )

if __name__ == "__main__":
    main()
