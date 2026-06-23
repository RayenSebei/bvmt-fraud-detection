"""
Phase 1.5 — Decay / Fading-Liquidity Detector.

The spike detector (anomaly_detector.py) catches the "Tuninvest pattern":
something leaked, volume/price jumped abnormally. It is the WRONG tool for
the "UADH pattern": a company quietly goes dark (stops publishing accounts,
stops holding AGOs, management becomes unreachable) and trading activity
fades toward nothing well before any official suspension. That's a decline,
not a spike — so it needs a different signal.

Two things computed per ticker:

  1. ACTIVITY DECLINE RATIO: total volume traded in the most recent 6
     months vs the 6 months before that. A ratio near 0 means trading
     has nearly stopped recently relative to its own (recent) history.
     We require a minimum baseline volume so we don't flag tickers that
     were always quiet — we want tickers that USED to trade and then
     stopped, which is the actual governance-vacuum signature.

  2. DAYS SINCE LAST ACTIVE: how long since the ticker last had any
     non-zero trading volume, measured from the dataset's most recent
     date. A large value means the stock has effectively gone silent
     and stayed silent.

This script also prints a "spotlight" — a month-by-month activity
timeline — for UADH specifically (the documented case), plus the top
3 most severe decliners found generally for side-by-side comparison
with the known March 2026 suspension.

Usage:
    python decay_detector.py
Reads bvmt_data/_all_tickers_combined.csv, writes:
    bvmt_data/decay_flags.csv
"""

import os

import pandas as pd

DATA_PATH = os.path.join("bvmt_data", "_all_tickers_combined.csv")
OUT_PATH = os.path.join("bvmt_data", "decay_flags.csv")

EXCLUDE_TICKERS = ["TBIDX"]  # index, not a stock
SPOTLIGHT_TICKER = "UADH"    # the documented case to specifically check

RECENT_MONTHS = 6
BASELINE_MONTHS = 6  # the 6 months immediately before the recent window
MIN_BASELINE_VOLUME = 1000  # ignore tickers that were already near-dead
DECLINE_RATIO_THRESHOLD = 0.2     # recent/baseline below this = flagged
DEAD_DAYS_THRESHOLD = 90          # no activity for this many days = flagged


def load_data(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[~df["symbole"].isin(EXCLUDE_TICKERS)].copy()
    df = df.sort_values(["symbole", "date"]).reset_index(drop=True)
    return df


def compute_decay_metrics(df):
    max_date = df["date"].max()
    recent_start = max_date - pd.DateOffset(months=RECENT_MONTHS)
    baseline_start = recent_start - pd.DateOffset(months=BASELINE_MONTHS)

    rows = []
    for symbole, g in df.groupby("symbole"):
        name = g["ticker_name"].iloc[0]

        recent = g[g["date"] >= recent_start]
        baseline = g[(g["date"] >= baseline_start) & (g["date"] < recent_start)]

        recent_volume = recent["volume"].sum()
        baseline_volume = baseline["volume"].sum()

        active_days = g[g["volume"] > 0]
        last_active_date = active_days["date"].max() if len(active_days) else pd.NaT
        days_since_active = (max_date - last_active_date).days if pd.notna(last_active_date) else None

        decline_ratio = (recent_volume / baseline_volume) if baseline_volume > 0 else None

        rows.append({
            "symbole": symbole,
            "ticker_name": name,
            "baseline_volume": baseline_volume,
            "recent_volume": recent_volume,
            "decline_ratio": decline_ratio,
            "last_active_date": last_active_date,
            "days_since_active": days_since_active,
        })

    return pd.DataFrame(rows)


def flag_decliners(metrics):
    eligible = metrics[metrics["baseline_volume"] >= MIN_BASELINE_VOLUME].copy()

    declining = eligible[
        (eligible["decline_ratio"].notna())
        & (eligible["decline_ratio"] < DECLINE_RATIO_THRESHOLD)
    ].copy()

    gone_dark = metrics[
        (metrics["days_since_active"].notna())
        & (metrics["days_since_active"] >= DEAD_DAYS_THRESHOLD)
    ].copy()

    flagged = pd.concat([declining, gone_dark]).drop_duplicates(subset=["symbole"])
    flagged = flagged.sort_values("decline_ratio", na_position="last")
    return flagged


def monthly_spotlight(df, symbole, months=18):
    g = df[df["symbole"] == symbole].copy()
    if g.empty:
        print(f"  No data found for {symbole} — check the ticker code is correct.")
        return

    g["month"] = g["date"].dt.to_period("M")
    monthly = g.groupby("month").agg(
        trading_days=("date", "count"),
        active_days=("volume", lambda s: (s > 0).sum()),
        total_volume=("volume", "sum"),
    ).reset_index()

    monthly = monthly.tail(months)
    print(f"\n  Month-by-month activity for {symbole} (last {months} months in data):")
    print(f"  {'Month':<10}{'TradingDays':<13}{'ActiveDays':<12}{'TotalVolume':<15}")
    for _, row in monthly.iterrows():
        print(f"  {str(row['month']):<10}{row['trading_days']:<13}{row['active_days']:<12}{row['total_volume']:<15.0f}")


if __name__ == "__main__":
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"  {df['symbole'].nunique()} tickers, {len(df)} rows")

    print("\nComputing decay metrics (recent 6mo vs prior 6mo activity)...")
    metrics = compute_decay_metrics(df)

    flagged = flag_decliners(metrics)
    flagged.to_csv(OUT_PATH, index=False)

    print(f"\n{len(flagged)} tickers flagged as declining/gone-dark.")
    print(f"Saved: {OUT_PATH}")

    print("\nTop 15 most severe decliners (lowest decline_ratio first):")
    cols = ["symbole", "ticker_name", "baseline_volume", "recent_volume",
            "decline_ratio", "days_since_active"]
    print(flagged[cols].head(15).to_string(index=False))

    print("\n" + "=" * 60)
    print(f"SPOTLIGHT: {SPOTLIGHT_TICKER} (documented case — suspended March 2026)")
    print("=" * 60)
    monthly_spotlight(df, SPOTLIGHT_TICKER, months=18)

    print("\n" + "=" * 60)
    print("SPOTLIGHT: Top 3 general decliners (excluding UADH if already shown)")
    print("=" * 60)
    top3 = flagged[flagged["symbole"] != SPOTLIGHT_TICKER].head(3)
    for _, row in top3.iterrows():
        monthly_spotlight(df, row["symbole"], months=18)

    print("\nWHAT TO LOOK FOR: a real 'going dark' case should show trading_days")
    print("and active_days fading toward 0 over consecutive months, NOT a single")
    print("quiet month surrounded by normal activity (that's just noise).")
