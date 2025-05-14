#!/usr/bin/env python3
"""
Streamlit dashboard to compare QQQ, QQ3.MI and a synthetic 3×-QQQ proxy.

Usage
-----
$ pip install -r requirements.txt
$ streamlit run app.py
"""
from __future__ import annotations

import logging
from datetime import date
from functools import lru_cache

import pandas as pd
import streamlit as st
import yfinance as yf

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="QQQ 3× ETF Comparison", layout="wide")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=8)
def _download(symbol: str, start: str) -> pd.DataFrame:
    """
    Download daily prices for *symbol* from *start* up to today.

    Args:
        symbol (str):
            Ticker (e.g. ``'QQQ'``).
        start (str):
            ISO-8601 start date (``'YYYY-MM-DD'``).

    Returns:
        pd.DataFrame:
            One column named exactly like *symbol*, *auto-adjusted* for splits
            and dividends. Example::

                >>> _download("QQQ", "2024-01-02").head()
                               QQQ
                Date
                2024-01-02  408.58

    Raises:
        ValueError: If no price column is found.
    """
    logging.info("Fetching %s from %s", symbol, start)
    df = yf.download(symbol, start=start, auto_adjust=True, progress=False)

    close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if close_col not in df.columns:
        raise ValueError(f"{symbol}: neither 'Close' nor 'Adj Close' in data.")

    return df[[close_col]].rename(columns={close_col: symbol}).dropna()


def build_dataset(start: str) -> pd.DataFrame:
    """
    Merge QQQ and QQ3.MI, then add a synthetic ``QQQ×3`` column.

    Args:
        start (str):
            ISO start date.

    Returns:
        pd.DataFrame:
            Columns ``['QQQ', 'QQ3.MI', 'QQQ×3']`` with common dates.

    Raises:
        ValueError: If the merge ends up empty (no overlapping dates).
    """
    qqq = _download("QQQ", start)
    qq3 = _download("QQ3.MI", start)

    merged = pd.concat([qqq, qq3], axis=1).dropna(how="all")
    merged["QQQ×3"] = merged["QQQ"] * 3
    merged = merged.dropna()

    if merged.empty:
        raise ValueError(
            "No overlapping data between QQQ and QQ3.MI for the selected start date."
        )

    logging.info("Merged dataset rows: %d", len(merged))
    return merged


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-index each column so the first valid observation = 100.

    Args:
        df (pd.DataFrame): Price DataFrame.

    Returns:
        pd.DataFrame: Re-based DataFrame or *df* unchanged if empty.
    """
    if df.empty:
        return df
    return df.div(df.iloc[0]).mul(100)


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.title("📈 Nasdaq-100 – 3× ETF Comparison")

    # Sidebar controls
    min_date = date(2000, 1, 1)  # QQQ inception
    start_date = st.sidebar.date_input(
        "Start date",
        value=date(2020, 1, 1),
        min_value=min_date,
        max_value=date.today(),
    )
    view_mode = st.sidebar.radio(
        "Display mode",
        options=("Raw price", "Normalised (start = 100)"),
    )

    # Fetch & transform data
    try:
        data = build_dataset(start_date.isoformat())
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    display_df = normalise(data) if "Normalised" in view_mode else data
    y_axis_label = (
        "Indexed level (start = 100)" if "Normalised" in view_mode else "Price"
    )

    # Charts & tables
    st.subheader("Daily series")
    st.line_chart(display_df, height=450)
    st.caption(y_axis_label)

    st.subheader("Latest snapshot")
    st.dataframe(
        display_df.tail(3).style.format("{:.2f}"), use_container_width=True
    )
    st.caption(
        "**QQQ×3** is a simple 3× multiple of QQQ (ignores daily compounding); "
        "**QQ3.MI** is the actual leveraged ETF traded on Borsa Italiana."
    )


if __name__ == "__main__":
    main()
