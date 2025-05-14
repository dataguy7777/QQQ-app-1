#!/usr/bin/env python3
"""
Streamlit dashboard to compare QQQ, QQ3.MI and a synthetic 3× QQQ proxy.

Run:
    pip install streamlit yfinance pandas
    streamlit run app.py
"""
from __future__ import annotations

import logging
from datetime import date
from functools import lru_cache

import pandas as pd
import streamlit as st
import yfinance as yf

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
st.set_page_config(page_title="QQQ 3× Comparison", layout="wide")


# --------------------------------------------------------------------------- #
# Data helpers                                                                #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8)
def _download(symbol: str, start: str) -> pd.DataFrame:
    """
    Downloads daily OHLC data for a given symbol with yfinance.

    Args:
        symbol (str): Ticker to download. Example: 'QQQ'
        start (str): ISO date 'YYYY-MM-DD'. Example: '2020-01-01'

    Returns:
        pd.DataFrame: Adj-Close series with DatetimeIndex. Example:

            >>> _download('QQQ', '2024-01-02').head()
                          Adj Close
            Date
            2024-01-02   408.579987
            2024-01-03   397.480011
    """
    logging.info("Fetching %s from %s", symbol, start)
    df = yf.download(symbol, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}.")
    return df[["Adj Close"]].rename(columns={"Adj Close": symbol})


def build_dataset(start: str) -> pd.DataFrame:
    """
    Builds a combined price DataFrame for QQQ, QQ3.MI and synthetic 3×QQQ.

    Args:
        start (str): ISO start date. Example: '2023-01-01'

    Returns:
        pd.DataFrame: Columns ['QQQ', 'QQ3.MI', 'QQQ×3']
    """
    qqq = _download("QQQ", start)
    qq3 = _download("QQ3.MI", start)
    merged = pd.concat([qqq, qq3], axis=1).dropna(how="all")
    merged["QQQ×3"] = merged["QQQ"] * 3
    merged = merged.dropna()
    logging.info("Merged dataset shape: %s", merged.shape)
    return merged


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-bases all series to 100 at the first observation.

    Args:
        df (pd.DataFrame): Price DataFrame

    Returns:
        pd.DataFrame: Re-based DataFrame
    """
    return df.div(df.iloc[0]).mul(100)


# --------------------------------------------------------------------------- #
# Streamlit UI                                                                #
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("📈 Nasdaq-100 3× ETF Comparison")

    min_date = date(2000, 1, 1)  # QQQ inception
    start_date = st.sidebar.date_input(
        "Start date", value=date(2020, 1, 1), min_value=min_date, max_value=date.today()
    )
    view_mode = st.sidebar.radio(
        "Display mode",
        options=("Raw price", "Normalised (start = 100)"),
        index=0,
    )

    try:
        data = build_dataset(start_date.isoformat())
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if view_mode.startswith("Normalised"):
        display_df = normalise(data)
        y_axis_label = "Indexed level (start = 100)"
    else:
        display_df = data
        y_axis_label = "Price (adjusted close)"

    st.subheader("Daily series")
    st.line_chart(display_df, height=450)
    st.caption(y_axis_label)

    st.subheader("Latest snapshot")
    st.dataframe(display_df.tail(3).style.format("{:.2f}"), use_container_width=True)
    st.caption(
        "Synthetic **QQQ×3** is simply 3 × QQQ (no compounding); "
        "**QQ3.MI** reflects the actual leveraged ETF traded in Milan."
    )


if __name__ == "__main__":
    main()
