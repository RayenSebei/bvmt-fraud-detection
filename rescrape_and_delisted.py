"""
Re-scrape gap-affected tickers + attempt fetching known DELISTED BVMT tickers.

Two jobs in one script:

1. RE-SCRAPE: tickers flagged with large unexplained gaps in the quality
   audit get re-downloaded from scratch with the new retry-and-log logic
   (see fetch_chunk_with_retry in scrape_ilboursa.py), so this time any
   chunk that fails gets printed explicitly instead of silently dropped.
   That should reveal whether the Dec-Mar gap pattern was a scraper bug
   (likely) or genuinely missing data.

2. DELISTED TICKERS: ilboursa's per-company pages (e.g. cotation_ADWYA)
   stay live after a company is delisted, so the /marches/download/{ticker}
   endpoint may still work even though the ticker no longer appears in the
   live dropdown. We test a small VERIFIED seed list of known delisted
   BVMT companies. This list is NOT exhaustive — BVMT went from 81 listed
   companies (Mar 2025) to 72 (early 2026), meaning more than a dozen
   names have likely delisted that aren't in this list yet. Add codes as
   you find them (search "radiée Bourse de Tunis" + year on ilboursa/Tustex).

"""

import os
from datetime import datetime

import pandas as pd
import requests

from scrape_ilboursa import (
    HEADERS,
    download_full_history,
)

OUT_DIR = "bvmt_data"
os.makedirs(OUT_DIR, exist_ok=True)

GAP_TICKERS = ["PLTU", "SIMPA", "SOTEM", "ALKIM", "SMD", "MIP", "WIFAK"]

KNOWN_DELISTED = [
    ("ADWYA", "ADWYA"),      # radiated Jan 2023 (OPR by Groupe Kilani/Teriak)
    ("AMS", "AMS"),          # radiated Sept 2023 (OPR)
    ("CEREALIS", "CEREALIS"),  # radiated, exact date unconfirmed — verify
]

START = datetime(2021, 6, 17)
END = datetime(2026, 6, 17)


def rescrape_gap_tickers(session):
    print("=" * 60)
    print("JOB 1: Re-scraping gap-affected tickers")
    print("=" * 60)

    all_failures = []
    for code in GAP_TICKERS:
        print(f"\nRe-downloading {code} (overwriting previous file)...")
        df, failed_ranges = download_full_history(code, START, END, session=session)

        if df.empty:
            print(f"  Still no data for {code} — needs manual check on ilboursa.com.")
            continue

        out_path = os.path.join(OUT_DIR, f"{code}.csv")
        df.insert(0, "ticker_name", code)
        df.to_csv(out_path, index=False)
        print(f"  Saved {len(df)} rows to {out_path}")

        if failed_ranges:
            print(f"  *** {len(failed_ranges)} chunk(s) still failed for {code} after retries:")
            for f_start, f_end in failed_ranges:
                print(f"      {f_start.date()} -> {f_end.date()}")
                all_failures.append((code, f_start.date(), f_end.date()))
        else:
            print(f"  No failed chunks this time — earlier gap was likely a transient scrape bug, now fixed.")

    return all_failures


def try_delisted_tickers(session):
    print("\n" + "=" * 60)
    print("JOB 2: Attempting known delisted tickers")
    print("=" * 60)

    recovered = []
    still_missing = []

    for code, name in KNOWN_DELISTED:
        print(f"\nTrying delisted ticker {code} ({name})...")
        try:
            df, failed_ranges = download_full_history(code, START, END, session=session)
        except Exception as e:
            print(f"  Request failed entirely for {code}: {e}")
            still_missing.append(code)
            continue

        if df.empty:
            print(f"  No data returned for {code} — endpoint likely doesn't serve "
                  f"delisted tickers, or the code is wrong. Verify the exact symbol "
                  f"on ilboursa.com/marches/cotation_{code}")
            still_missing.append(code)
            continue

        out_path = os.path.join(OUT_DIR, f"{code}.csv")
        df.insert(0, "ticker_name", name)
        df.to_csv(out_path, index=False)
        print(f"  SUCCESS — saved {len(df)} rows to {out_path} "
              f"(date range {df['date'].min().date()} -> {df['date'].max().date()})")
        recovered.append(code)

        if failed_ranges:
            print(f"  ({len(failed_ranges)} chunk(s) failed within this ticker's range)")

    return recovered, still_missing


if __name__ == "__main__":
    session = requests.Session()
    session.headers.update(HEADERS)

    gap_failures = rescrape_gap_tickers(session)
    recovered, still_missing = try_delisted_tickers(session)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if gap_failures:
        print(f"\n{len(gap_failures)} chunk(s) STILL failing after retry across gap tickers:")
        for code, f_start, f_end in gap_failures:
            print(f"  {code}: {f_start} -> {f_end}")
        print("These are likely genuine missing data on ilboursa's side, not a scraper bug.")
    else:
        print("\nAll previously-gapped tickers re-downloaded cleanly with no failed chunks.")
        print("This confirms the original gaps were a scraper bug (now fixed), not a real market closure.")

    if recovered:
        print(f"\nRecovered delisted tickers: {', '.join(recovered)}")
        print("These are now in bvmt_data/ and should be INCLUDED in your factor universe")
        print("to avoid survivorship bias.")

    if still_missing:
        print(f"\nCould not recover: {', '.join(still_missing)}")
        print("Try manually downloading via the website for these, or confirm the exact ticker code.")

