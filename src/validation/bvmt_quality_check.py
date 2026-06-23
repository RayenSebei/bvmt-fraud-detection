"""
BVMT (_all_tickers_combined.csv) Full Data Quality Check
=========================================================
1. Structure        - row count, unique tickers, column dtypes
2. Coverage         - min/max date and row count per ticker; flags < 1 year or < 50 rows
3. Duplicates       - (symbole, date) duplicate pairs
4. Missing values   - NaN counts per column; tickers with price/volume NaNs
5. OHLC consistency - haut >= bas/open/close; bas <= open/close
6. Volume sanity    - negative volume; zero-volume % per ticker (flag > 50%)
7. Price outliers   - daily close-to-close change > +-30%
8. Date gaps        - consecutive trading-day gaps > 10 calendar days
9. Final summary    - tickers / rows needing manual review
"""

import pandas as pd
import numpy as np
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parents[2] / "bvmt_data" / "_all_tickers_combined.csv"
SEP = "=" * 72


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
print(f"Loading: {CSV_PATH}")
df_raw = pd.read_csv(CSV_PATH)
print(f"  Raw shape: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")


# ─────────────────────────────────────────────
# 1. STRUCTURE
# ─────────────────────────────────────────────
section("1. STRUCTURE")

print(f"\n  Total rows        : {len(df_raw):,}")
print(f"  Total columns     : {len(df_raw.columns)}")
print(f"  Columns           : {list(df_raw.columns)}")
print(f"  Unique tickers    : {df_raw['symbole'].nunique()}")

print("\n  Raw dtype check (before coercion):")
for col, dt in df_raw.dtypes.items():
    note = ""
    if col == "date" and str(dt) != "datetime64[ns]":
        note = "  [WARNING] should be datetime"
    elif col in ("ouverture", "haut", "bas", "cloture") and not pd.api.types.is_float_dtype(dt):
        note = "  [WARNING] should be float"
    elif col == "volume" and not pd.api.types.is_integer_dtype(dt):
        note = "  [WARNING] should be int"
    print(f"    {col:<12} {str(dt):<16} {note}")

# Coerce types
df = df_raw.copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
for col in ("ouverture", "haut", "bas", "cloture"):
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

date_parse_failures = df["date"].isna().sum()
if date_parse_failures:
    print(f"\n  [WARNING] {date_parse_failures} rows could not be parsed as dates (set to NaT)")
else:
    print("\n  [OK] All dates parsed successfully")

print("\n  Dtypes after coercion:")
for col, dt in df.dtypes.items():
    print(f"    {col:<12} {str(dt)}")


# ─────────────────────────────────────────────
# 2. COVERAGE PER TICKER
# ─────────────────────────────────────────────
section("2. COVERAGE PER TICKER")

coverage = (
    df.groupby("symbole")["date"]
    .agg(min_date="min", max_date="max", row_count="count")
    .reset_index()
)
coverage["date_range_days"] = (coverage["max_date"] - coverage["min_date"]).dt.days
coverage["years"] = coverage["date_range_days"] / 365.25

flagged_coverage = coverage[
    (coverage["date_range_days"] < 365) | (coverage["row_count"] < 50)
].copy()

print(f"\n  {'Ticker':<10} {'Min Date':<12} {'Max Date':<12} {'Rows':>6} {'Years':>6}  Status")
print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*6} {'-'*6}  ------")
for _, r in coverage.sort_values("symbole").iterrows():
    flags = []
    if r["date_range_days"] < 365:
        flags.append("< 1yr range")
    if r["row_count"] < 50:
        flags.append("< 50 rows")
    flag_str = "  [FLAG] " + ", ".join(flags) if flags else "  [OK]"
    print(
        f"  {r['symbole']:<10} {str(r['min_date'].date()):<12} "
        f"{str(r['max_date'].date()):<12} {r['row_count']:>6} {r['years']:>6.1f}{flag_str}"
    )

print(f"\n  Flagged tickers (coverage issues): {len(flagged_coverage)}")
if not flagged_coverage.empty:
    print("  -> " + ", ".join(flagged_coverage["symbole"].tolist()))


# ─────────────────────────────────────────────
# 3. DUPLICATES
# ─────────────────────────────────────────────
section("3. DUPLICATE (symbole, date) PAIRS")

dup_mask = df.duplicated(subset=["symbole", "date"], keep=False)
dup_df = df[dup_mask].copy()
n_dup_rows = len(dup_df)
n_dup_pairs = df[df.duplicated(subset=["symbole", "date"], keep="first")].shape[0]

print(f"\n  Duplicate rows  : {n_dup_rows:,}")
print(f"  Duplicate pairs : {n_dup_pairs:,}")
if n_dup_pairs > 0:
    print("\n  Affected tickers:")
    for ticker, cnt in dup_df["symbole"].value_counts().items():
        print(f"    {ticker:<10} {cnt} duplicated rows")
    print("\n  Sample duplicates (up to 10):")
    print(dup_df[["symbole", "date", "ouverture", "cloture", "volume"]].head(10).to_string(index=True))
else:
    print("  [OK] No duplicate (symbole, date) pairs found")


# ─────────────────────────────────────────────
# 4. MISSING VALUES
# ─────────────────────────────────────────────
section("4. MISSING VALUES")

nan_counts = df.isnull().sum()
print("\n  NaN counts per column:")
for col, cnt in nan_counts.items():
    marker = "  [WARN]" if cnt > 0 else "  [OK]  "
    print(f"  {marker} {col:<12} {cnt:>6} NaNs  ({cnt / len(df) * 100:.2f}%)")

price_vol_cols = ["ouverture", "haut", "bas", "cloture", "volume"]
nan_in_prices = df[df[price_vol_cols].isnull().any(axis=1)][["symbole", "date"] + price_vol_cols]

if len(nan_in_prices) > 0:
    print(f"\n  [WARNING] {len(nan_in_prices)} rows have NaN in price/volume columns")
    print("  Tickers with price/volume NaNs:")
    for ticker, cnt in nan_in_prices["symbole"].value_counts().items():
        print(f"    {ticker:<10} {cnt} row(s)")
    print("\n  Sample rows (up to 10):")
    print(nan_in_prices.head(10).to_string(index=True))
else:
    print("\n  [OK] No NaNs in price/volume columns")


# ─────────────────────────────────────────────
# 5. OHLC CONSISTENCY
# ─────────────────────────────────────────────
section("5. OHLC CONSISTENCY")

df_valid = df.dropna(subset=["ouverture", "haut", "bas", "cloture"]).copy()

violations = {
    "haut < bas"      : df_valid["haut"] < df_valid["bas"],
    "haut < ouverture": df_valid["haut"] < df_valid["ouverture"],
    "haut < cloture"  : df_valid["haut"] < df_valid["cloture"],
    "bas > ouverture" : df_valid["bas"]  > df_valid["ouverture"],
    "bas > cloture"   : df_valid["bas"]  > df_valid["cloture"],
}

any_violation = pd.Series(False, index=df_valid.index)
for rule, mask in violations.items():
    any_violation = any_violation | mask

ohlc_bad = df_valid[any_violation].copy()

print(f"\n  OHLC rule violations:")
for rule, mask in violations.items():
    print(f"    {rule:<22}: {mask.sum():>4} row(s)")

print(f"\n  Total rows with >= 1 OHLC violation: {len(ohlc_bad)}")
if len(ohlc_bad) > 0:
    print(f"\n  Affected tickers: {sorted(ohlc_bad['symbole'].unique().tolist())}")
    print("\n  Violating rows (up to 20):")
    print(
        ohlc_bad[["symbole", "date", "ouverture", "haut", "bas", "cloture"]]
        .head(20)
        .to_string(index=True)
    )
else:
    print("  [OK] All OHLC relationships are consistent")


# ─────────────────────────────────────────────
# 6. VOLUME SANITY
# ─────────────────────────────────────────────
section("6. VOLUME SANITY")

df_vol = df.dropna(subset=["volume"]).copy()
neg_vol = df_vol[df_vol["volume"] < 0]
print(f"\n  Rows with negative volume: {len(neg_vol)}")
if len(neg_vol) > 0:
    print(neg_vol[["symbole", "date", "volume"]].to_string(index=True))
else:
    print("  [OK] No negative volume rows")

vol_stats = df_vol.groupby("symbole").agg(
    total_rows=("volume", "count"),
    zero_vol_rows=("volume", lambda x: (x == 0).sum()),
).reset_index()
vol_stats["zero_pct"] = vol_stats["zero_vol_rows"] / vol_stats["total_rows"] * 100
vol_stats_flagged = vol_stats[vol_stats["zero_pct"] > 50].sort_values("zero_pct", ascending=False)

print(f"\n  Zero-volume day statistics (flagging > 50%):")
print(f"  {'Ticker':<10} {'Total Rows':>10} {'Zero-Vol Rows':>14} {'Zero %':>8}  Status")
print(f"  {'-'*10} {'-'*10} {'-'*14} {'-'*8}  ------")
for _, r in vol_stats.sort_values("zero_pct", ascending=False).iterrows():
    marker = "  [FLAG]" if r["zero_pct"] > 50 else "  [OK] "
    print(
        f"{marker} {r['symbole']:<10} {r['total_rows']:>10,} "
        f"{r['zero_vol_rows']:>14,} {r['zero_pct']:>7.1f}%"
    )

print(f"\n  Tickers with > 50% zero-volume days: {len(vol_stats_flagged)}")
if not vol_stats_flagged.empty:
    print("  -> " + ", ".join(vol_stats_flagged["symbole"].tolist()))


# ─────────────────────────────────────────────
# 7. PRICE OUTLIERS (close-to-close > +-30%)
# ─────────────────────────────────────────────
section("7. PRICE OUTLIERS (daily close-to-close > +-30%)")

df_sorted = df.dropna(subset=["cloture", "date"]).sort_values(["symbole", "date"]).copy()
df_sorted["prev_close"] = df_sorted.groupby("symbole")["cloture"].shift(1)
df_sorted["pct_change"] = (
    (df_sorted["cloture"] - df_sorted["prev_close"]) / df_sorted["prev_close"] * 100
)

outliers = df_sorted[df_sorted["pct_change"].abs() > 30].copy()

print(f"\n  Rows with |daily close change| > 30%: {len(outliers)}")
if len(outliers) > 0:
    print(f"\n  {'Ticker':<10} {'Date':<12} {'Prev Close':>10} {'Close':>10} {'Change %':>10}")
    print(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for _, r in outliers.sort_values("pct_change", key=abs, ascending=False).iterrows():
        print(
            f"  {r['symbole']:<10} {str(r['date'].date()):<12} "
            f"{r['prev_close']:>10.3f} {r['cloture']:>10.3f} {r['pct_change']:>+10.1f}%"
        )
else:
    print("  [OK] No extreme daily price moves detected")


# ─────────────────────────────────────────────
# 8. DATE GAPS (> 10 calendar days)
# ─────────────────────────────────────────────
section("8. DATE GAPS (consecutive gap > 10 calendar days)")

df_dates = df.dropna(subset=["date"]).sort_values(["symbole", "date"]).copy()
df_dates["prev_date"] = df_dates.groupby("symbole")["date"].shift(1)
df_dates["gap_days"] = (df_dates["date"] - df_dates["prev_date"]).dt.days

large_gaps = df_dates[df_dates["gap_days"] > 10].copy()

print(f"\n  Gaps > 10 calendar days: {len(large_gaps)}")
if len(large_gaps) > 0:
    print(f"\n  {'Ticker':<10} {'From Date':<12} {'To Date':<12} {'Gap (days)':>10}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
    for _, r in large_gaps.sort_values("gap_days", ascending=False).iterrows():
        print(
            f"  {r['symbole']:<10} {str(r['prev_date'].date()):<12} "
            f"{str(r['date'].date()):<12} {int(r['gap_days']):>10}"
        )
else:
    print("  [OK] No large date gaps detected")

gap_tickers = sorted(large_gaps["symbole"].unique().tolist())


# ─────────────────────────────────────────────
# 9. FINAL SUMMARY
# ─────────────────────────────────────────────
section("9. FINAL SUMMARY -- TICKERS NEEDING MANUAL REVIEW")

review_reasons = {}


def flag(ticker, reason):
    review_reasons.setdefault(ticker, []).append(reason)


# Coverage
for _, r in flagged_coverage.iterrows():
    reasons = []
    if r["date_range_days"] < 365:
        reasons.append(f"short date range ({r['date_range_days']} days)")
    if r["row_count"] < 50:
        reasons.append(f"few rows ({r['row_count']})")
    flag(r["symbole"], " + ".join(reasons))

# Duplicates
if n_dup_pairs > 0:
    for ticker in dup_df["symbole"].unique():
        flag(ticker, "duplicate (symbole, date) rows")

# NaN in price/volume
if len(nan_in_prices) > 0:
    for ticker in nan_in_prices["symbole"].unique():
        flag(ticker, "NaN in price/volume column(s)")

# OHLC violations
if len(ohlc_bad) > 0:
    for ticker in ohlc_bad["symbole"].unique():
        flag(ticker, "OHLC consistency violation")

# Negative volume
if len(neg_vol) > 0:
    for ticker in neg_vol["symbole"].unique():
        flag(ticker, "negative volume row(s)")

# High zero-volume
for _, r in vol_stats_flagged.iterrows():
    flag(r["symbole"], f"high zero-volume ({r['zero_pct']:.0f}% of days)")

# Price outliers
if len(outliers) > 0:
    for ticker in outliers["symbole"].unique():
        flag(ticker, "extreme daily price move (> +-30%)")

# Date gaps
for ticker in gap_tickers:
    flag(ticker, "large date gap (> 10 calendar days)")

print(f"\n  Tickers requiring manual review: {len(review_reasons)}")
if review_reasons:
    print(f"\n  {'Ticker':<10}  Issues")
    print(f"  {'-'*10}  {'-'*60}")
    for ticker in sorted(review_reasons.keys()):
        print(f"  {ticker:<10}  {' | '.join(review_reasons[ticker])}")
else:
    print("  [OK] No tickers flagged -- dataset looks clean!")

# Row-level total
rows_to_review = set()
if n_dup_pairs > 0:
    rows_to_review.update(dup_df.index.tolist())
if len(nan_in_prices) > 0:
    rows_to_review.update(nan_in_prices.index.tolist())
if len(ohlc_bad) > 0:
    rows_to_review.update(ohlc_bad.index.tolist())
if len(neg_vol) > 0:
    rows_to_review.update(neg_vol.index.tolist())
if len(outliers) > 0:
    rows_to_review.update(outliers.index.tolist())
if len(large_gaps) > 0:
    rows_to_review.update(large_gaps.index.tolist())

pct = len(rows_to_review) / len(df) * 100 if len(df) > 0 else 0
print(
    f"\n  Total flagged rows (across all checks): "
    f"{len(rows_to_review):,} / {len(df):,}  ({pct:.2f}%)"
)

print(f"\n{SEP}")
print("  Quality check complete.")
print(SEP)
