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

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="QQQ 3× ETF Comparison", layout="wide")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

TICKER_BASE = "QQQ"        # Nasdaq-100 ETF
TICKER_LEV  = "QQQ3.MI"    # 3× leveraged Nasdaq-100 ETF (Borsa Italiana)

# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=8)
def _download(symbol: str, start: str) -> pd.DataFrame:
    """
    Fetch daily adjusted prices for *symbol* starting from *start*.

    Returns
    -------
    pd.DataFrame
        One tz-naive column named *symbol*.

    Raises
    ------
    ValueError
        If price column not found.
    """
    logging.info("Fetching %s from %s", symbol, start)
    df = yf.download(symbol, start=start, auto_adjust=True, progress=False)

    close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if close_col not in df.columns:
        raise ValueError(f"{symbol}: neither 'Close' nor 'Adj Close' present.")

    if df.index.tz is not None:                      # drop timezone → tz-naive
        df.index = df.index.tz_convert(None)

    return df[[close_col]].rename(columns={close_col: symbol}).dropna()


def build_dataset(start: str) -> pd.DataFrame:
    """
    Merge QQQ and QQQ3.MI, then add synthetic ``QQQ×3``.

    Raises
    ------
    ValueError
        If no overlapping dates.
    """
    qqq  = _download(TICKER_BASE, start)
    qqq3 = _download(TICKER_LEV,  start)

    merged = pd.concat([qqq, qqq3], axis=1).dropna(how="all")
    merged["QQQ×3"] = merged[TICKER_BASE] * 3
    merged = merged.dropna()

    if merged.empty:
        raise ValueError("No overlapping data for the selected start date.")

    logging.info("Merged rows: %d", len(merged))
    return merged


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Re-base all columns so the first valid point = 100."""
    if df.empty:
        return df
    return df.div(df.iloc[0]).mul(100)


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.title("📈 Nasdaq-100 – 3× ETF Comparison")

    # Sidebar
    min_date = date(2000, 1, 1)
    start_date = st.sidebar.date_input(
        "Start date",
        value=date(2020, 1, 1),
        min_value=min_date,
        max_value=date.today(),
    )
    view_mode = st.sidebar.radio(
        "Display mode", ("Raw price", "Normalised (start = 100)")
    )

    # Data
    try:
        data = build_dataset(start_date.isoformat())
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    display_df = normalise(data) if "Normalised" in view_mode else data
    y_label = "Indexed level (start = 100)" if "Normalised" in view_mode else "Price"

    # ---- Chart ----------------------------------------------------------------
    chart_df = (
        display_df
        .reset_index()                       # make the DatetimeIndex a column
        .rename(columns={"index": "Date"})   # ensure the column is named Date
        .melt(id_vars="Date", var_name="Ticker", value_name="Price")
    )

    line_chart = (
        alt.Chart(chart_df)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Price:Q", title=y_label),
            color=alt.Color("Ticker:N", title=""),
            tooltip=["Date:T", "Ticker:N", "Price:Q"],
        )
        .properties(height=450)
        .interactive()
    )

    st.altair_chart(line_chart, use_container_width=True)

    # ---- Latest snapshot ------------------------------------------------------
    st.subheader("Latest snapshot")
    st.dataframe(display_df.tail(3).style.format("{:.2f}"), use_container_width=True)
    st.caption(
        "**QQQ×3** is a simple 3× multiple of QQQ (ignores daily compounding); "
        "**QQQ3.MI** is the real 3× ETF listed on Borsa Italiana."
    )


if __name__ == "__main__":
    main()
