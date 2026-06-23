"""
Phase 1 — Volume & Price Anomaly Detector for BVMT market surveillance.

This is NOT a fraud detector in the legal sense — it's a statistical
screening tool, the same kind of approach real exchanges/regulators use
(SEC, ESMA, FINRA market surveillance systems): flag abnormal patterns
that historically correlate with manipulation/insider-trading cases, so
a human can investigate further. It does not prove fraud on its own.

Two signals computed per ticker, per day:
  1. Volume anomaly: rolling z-score of volume vs trailing baseline.
     A spike means trading activity far outside the norm for that stock.
  2. Price-volume divergence: large single-day return co-occurring with
     a volume anomaly. This is the classic "something happened that the
     market knew about before it was public" pattern.

IMPORTANT CAVEAT: without disclosure/news timestamps (Phase 2), this
script cannot tell the difference between:
  - a real red flag (info leaked before public announcement), and
  - a routine, fully-explained event (rights issue, dividend, split).
So treat every flagged row as "needs manual research," not "confirmed
anomaly." Phase 2 will cross-reference BVMT's avis-décisions bulletin
to auto-distinguish explained vs unexplained spikes.

Usage:
    python src/detection/anomaly_detector.py
Reads bvmt_data/_all_tickers_combined.csv, writes:
    bvmt_data/anomaly_flags.csv      (every flagged ticker/date)
    bvmt_data/anomaly_summary.csv    (one row per ticker, flag counts)
"""

import os

import numpy as np
import pandas as pd

DATA_PATH = os.path.join("bvmt_data", "_all_tickers_combined.csv")
OUT_FLAGS = os.path.join("bvmt_data", "anomaly_flags.csv")
OUT_SUMMARY = os.path.join("bvmt_data", "anomaly_summary.csv")

# Tickers to exclude from analysis entirely
EXCLUDE_TICKERS = ["TBIDX", "PLTU"]  # index + known-incomplete ticker

# Tuning parameters — these are reasonable starting points, not gospel.
# Tighten/loosen after seeing how many flags come out.
VOLUME_ROLLING_WINDOW = 60     # trading days for the volume baseline
VOLUME_Z_THRESHOLD = 3.0       # how many std devs above normal = anomaly
RETURN_Z_THRESHOLD = 3.0       # how many std devs of daily return = large move
MIN_HISTORY_DAYS = 60          # don't flag anything until baseline is established


def load_data(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[~df["symbole"].isin(EXCLUDE_TICKERS)].copy()

    # Fix the rounding-error OHLC violations: clamp close into [low, high]
    df["cloture"] = df[["cloture", "bas"]].max(axis=1)
    df["cloture"] = df[["cloture", "haut"]].min(axis=1)

    df = df.sort_values(["symbole", "date"]).reset_index(drop=True)
    return df


def compute_signals(df):
    results = []

    for symbole, g in df.groupby("symbole"):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < MIN_HISTORY_DAYS:
            continue  # too little history to establish a reliable baseline

        g["daily_return"] = g["cloture"].pct_change()

        # Volume z-score vs trailing rolling window (excludes current day
        # from its own baseline via shift(1), avoiding look-ahead/self-bias)
        roll_mean = g["volume"].shift(1).rolling(VOLUME_ROLLING_WINDOW, min_periods=20).mean()
        roll_std = g["volume"].shift(1).rolling(VOLUME_ROLLING_WINDOW, min_periods=20).std()
        g["volume_zscore"] = (g["volume"] - roll_mean) / roll_std.replace(0, np.nan)

        # Return z-score vs trailing rolling window of returns
        ret_roll_mean = g["daily_return"].shift(1).rolling(VOLUME_ROLLING_WINDOW, min_periods=20).mean()
        ret_roll_std = g["daily_return"].shift(1).rolling(VOLUME_ROLLING_WINDOW, min_periods=20).std()
        g["return_zscore"] = (g["daily_return"] - ret_roll_mean) / ret_roll_std.replace(0, np.nan)

        g["volume_anomaly"] = g["volume_zscore"] > VOLUME_Z_THRESHOLD
        g["price_anomaly"] = g["return_zscore"].abs() > RETURN_Z_THRESHOLD
        g["combined_anomaly"] = g["volume_anomaly"] & g["price_anomaly"]

        results.append(g)

    return pd.concat(results, ignore_index=True)


def summarize(df):
    flagged = df[df["volume_anomaly"] | df["price_anomaly"]].copy()

    summary = (
        flagged.groupby("symbole")
        .agg(
            ticker_name=("ticker_name", "first"),
            volume_anomalies=("volume_anomaly", "sum"),
            price_anomalies=("price_anomaly", "sum"),
            combined_anomalies=("combined_anomaly", "sum"),
            first_flag=("date", "min"),
            last_flag=("date", "max"),
        )
        .reset_index()
        .sort_values("combined_anomalies", ascending=False)
    )
    return flagged, summary


if __name__ == "__main__":
    print("Loading and cleaning data...")
    df = load_data(DATA_PATH)
    print(f"  {df['symbole'].nunique()} tickers, {len(df)} rows after exclusions")

    print("\nComputing volume and price anomaly signals...")
    df = compute_signals(df)

    print("\nSummarizing flagged events...")
    flagged, summary = summarize(df)

    flagged_out = flagged[[
        "symbole", "ticker_name", "date", "cloture", "volume",
        "daily_return", "volume_zscore", "return_zscore",
        "volume_anomaly", "price_anomaly", "combined_anomaly",
    ]]
    flagged_out.to_csv(OUT_FLAGS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    print(f"\n{len(flagged)} flagged rows across {summary.shape[0]} tickers.")
    print(f"  {flagged['combined_anomaly'].sum()} are COMBINED anomalies "
          f"(both volume spike + large price move same day — highest priority).")
    print(f"\nSaved: {OUT_FLAGS}")
    print(f"Saved: {OUT_SUMMARY}")

    print("\nTop 10 tickers by combined anomaly count:")
    print(summary.head(10).to_string(index=False))

    print("\n--- VALIDATION CHECK ---")
    print("Search for 'TINV' or Tuninvest's actual ticker code in anomaly_flags.csv")
    print("for the Oct-Nov 2025 window. If flagged there, the detector correctly")
    print("caught the real documented case — strong validation signal.")
