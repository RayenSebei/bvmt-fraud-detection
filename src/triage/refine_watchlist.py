"""
Index cross-check refinement — distinguish market-wide moves from genuine
company-specific anomalies in the watchlist.

A stock spiking on the same day as several other stocks, or on a day the
whole Tunindex moved sharply, is very likely reacting to a market-wide
event (macro news, rate decision, broad rally/selloff) rather than hiding
company-specific information. The per-company news search built earlier
can't catch this on its own, since it only checks each company's own
headlines, not market-wide or index-level news.

Adds three things to every watchlist row:
  1. index_return / index_return_zscore — how Tunindex (PX1) itself moved
     that day, using the same rolling z-score method as anomaly_detector.py
  2. co_flagged_count — how many OTHER tickers were also flagged (spike
     detector) on that same date
  3. excess_return — the stock's own return minus the index's return that
     day (spike_detector rows only). A large excess return even after
     removing the market-wide component is the real signature of a
     genuinely company-specific move.

Tags each row MARKET_WIDE or COMPANY_SPECIFIC and writes a sharper,
re-prioritized watchlist.

Reads:
    bvmt_data/_all_tickers_combined.csv  (for the Tunindex/PX1 series)
    bvmt_data/anomaly_flags.csv          (for co-flagged ticker counts)
    bvmt_data/watchlist.csv
Writes:
    bvmt_data/watchlist_refined.csv

Usage:
    python refine_watchlist.py
"""

import os

import numpy as np
import pandas as pd

OUT_DIR = "bvmt_data"
COMBINED_PATH = os.path.join(OUT_DIR, "_all_tickers_combined.csv")
ANOMALY_FLAGS_PATH = os.path.join(OUT_DIR, "anomaly_flags.csv")
WATCHLIST_PATH = os.path.join(OUT_DIR, "watchlist.csv")
OUT_PATH = os.path.join(OUT_DIR, "watchlist_refined.csv")

INDEX_TICKER = "PX1"  # TUNINDEX, confirmed from the ticker dropdown
ROLLING_WINDOW = 60
INDEX_ZSCORE_THRESHOLD = 2.0   # index itself moved unusually that day
CO_FLAG_THRESHOLD = 3          # this many OTHER tickers also flagged same day


def build_index_series(combined_path, index_ticker=INDEX_TICKER):
    df = pd.read_csv(combined_path)
    df["date"] = pd.to_datetime(df["date"])
    idx = df[df["symbole"] == index_ticker].copy()
    if idx.empty:
        raise ValueError(f"No data found for index ticker '{index_ticker}' in {combined_path}")

    idx = idx.sort_values("date").reset_index(drop=True)
    idx["index_return"] = idx["cloture"].pct_change()

    roll_mean = idx["index_return"].shift(1).rolling(ROLLING_WINDOW, min_periods=20).mean()
    roll_std = idx["index_return"].shift(1).rolling(ROLLING_WINDOW, min_periods=20).std()
    idx["index_return_zscore"] = (idx["index_return"] - roll_mean) / roll_std.replace(0, np.nan)

    return idx.set_index("date")[["index_return", "index_return_zscore"]]


def build_co_flagged_counts(anomaly_flags_path):
    df = pd.read_csv(anomaly_flags_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby("date")["symbole"].nunique()  # distinct tickers flagged per date


def refine_watchlist(watchlist_path, index_series, co_flag_counts):
    wl = pd.read_csv(watchlist_path)
    wl["date"] = pd.to_datetime(wl["date"], errors="coerce")

    index_returns, index_zscores = [], []
    co_counts, excess_returns, tags = [], [], []

    for _, row in wl.iterrows():
        date = row["date"]

        if pd.isna(date):
            # decay_detector rows have no single date — pass through untagged
            index_returns.append(None)
            index_zscores.append(None)
            co_counts.append(None)
            excess_returns.append(None)
            tags.append("N/A (no single date)")
            continue

        idx_ret = index_series["index_return"].get(date, None)
        idx_z = index_series["index_return_zscore"].get(date, None)

        total_flagged_that_day = co_flag_counts.get(date, 1)
        co_count = max(total_flagged_that_day - 1, 0)  # exclude the row's own ticker

        stock_ret = row.get("daily_return", None)
        excess_ret = None
        if stock_ret is not None and idx_ret is not None and pd.notna(stock_ret) and pd.notna(idx_ret):
            excess_ret = stock_ret - idx_ret

        is_market_wide = (
            (idx_z is not None and pd.notna(idx_z) and abs(idx_z) >= INDEX_ZSCORE_THRESHOLD)
            or (co_count >= CO_FLAG_THRESHOLD)
        )
        tag = "MARKET_WIDE" if is_market_wide else "COMPANY_SPECIFIC"

        index_returns.append(idx_ret)
        index_zscores.append(idx_z)
        co_counts.append(co_count)
        excess_returns.append(excess_ret)
        tags.append(tag)

    wl["index_return"] = index_returns
    wl["index_return_zscore"] = index_zscores
    wl["co_flagged_count"] = co_counts
    wl["excess_return"] = excess_returns
    wl["tag"] = tags

    return wl


if __name__ == "__main__":
    print("Building Tunindex (PX1) reference series...")
    index_series = build_index_series(COMBINED_PATH)
    print(f"  {len(index_series)} index trading days loaded")

    print("\nCounting co-flagged tickers per date...")
    co_flag_counts = build_co_flagged_counts(ANOMALY_FLAGS_PATH)

    print("\nRefining watchlist...")
    refined = refine_watchlist(WATCHLIST_PATH, index_series, co_flag_counts)
    refined.to_csv(OUT_PATH, index=False)

    n_total = len(refined)
    n_market_wide = (refined["tag"] == "MARKET_WIDE").sum()
    n_company_specific = (refined["tag"] == "COMPANY_SPECIFIC").sum()
    n_na = (refined["tag"] == "N/A (no single date)").sum()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{n_total} watchlist rows refined:")
    print(f"  MARKET_WIDE (likely broad market move, deprioritize): {n_market_wide}")
    print(f"  COMPANY_SPECIFIC (genuinely isolated, top priority):  {n_company_specific}")
    print(f"  N/A (decay-detector rows, no single date):            {n_na}")

    print("\nTop COMPANY_SPECIFIC anomalies (sorted by |excess_return|, most isolated first):")
    specific = refined[refined["tag"] == "COMPANY_SPECIFIC"].copy()
    if "excess_return" in specific.columns:
        specific["abs_excess"] = specific["excess_return"].abs()
        specific = specific.sort_values("abs_excess", ascending=False)

    cols = [c for c in [
        "symbole", "ticker_name", "date", "volume_zscore", "return_zscore",
        "index_return_zscore", "co_flagged_count", "excess_return",
    ] if c in specific.columns]
    print(specific[cols].head(25).to_string(index=False))

    print(f"\nSaved: {OUT_PATH}")
    print("\nFocus manual research on COMPANY_SPECIFIC rows first — these are isolated,")
    print("single-stock moves with no broad market move or news explanation behind them.")
