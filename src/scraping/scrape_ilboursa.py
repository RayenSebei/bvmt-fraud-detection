"""
Scrape historical OHLCV data from ilboursa.com for BVMT-listed tickers.

ilboursa enforces a 3-month max date range per download AND requires
a CSRF token (__RequestVerificationToken) that's tied to the session
cookie set when you GET the page. So for each chunk we:
    1. GET the download page to refresh cookies + extract a fresh token
    2. POST the date range + token to get the CSV back
    3. Repeat for each 3-month window, then concatenate
"""

import io
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.ilboursa.com/marches/download/{ticker}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
}


def get_csrf_token(session, ticker):
    """GET the download page and extract the current CSRF token."""
    url = BASE_URL.format(ticker=ticker)
    resp = session.get(url, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    token_input = soup.find("input", {"name": "__RequestVerificationToken"})
    if token_input is None:
        raise RuntimeError(
            f"No CSRF token found for {ticker} — page structure may differ "
            "from what was inspected, or the ticker code is wrong."
        )
    return token_input["value"]


def fetch_chunk(session, ticker, date_from, date_to):
    """POST one <=3-month date range and return the raw response text (CSV expected)."""
    url = BASE_URL.format(ticker=ticker)
    token = get_csrf_token(session, ticker)

    payload = {
        "dtFrom": date_from.strftime("%Y-%m-%d"),
        "dtTo": date_to.strftime("%Y-%m-%d"),
        "__RequestVerificationToken": token,
    }

    resp = session.post(url, data=payload, headers={"Referer": url}, timeout=15)
    resp.raise_for_status()
    return resp.text


def discover_tickers(session, sample_ticker="PX1"):
    """
    Try to auto-discover the full list of ticker codes from the
    'Choisir une valeur' dropdown present on the download page.
    Returns a list of (code, name) tuples. If this fails (site structure
    differs from expected), it prints the <select> elements it found for
    manual inspection.
    """
    url = BASE_URL.format(ticker=sample_ticker)
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    selects = soup.find_all("select")
    best = None
    for sel in selects:
        opts = sel.find_all("option")
        if len(opts) > 50:  # the ticker dropdown should have ~80 entries
            best = sel
            break

    if best is None:
        print("Could not auto-find the ticker dropdown. Found these <select> tags instead:")
        for sel in selects:
            print(f"  id={sel.get('id')!r} name={sel.get('name')!r} options={len(sel.find_all('option'))}")
        return []

    tickers = []
    for opt in best.find_all("option"):
        code = opt.get("value", "").strip()
        name = opt.get_text(strip=True)
        if code:
            tickers.append((code, name))
    return tickers


def parse_csv_chunk(text):
    """Parse one raw semicolon-delimited, comma-decimal CSV chunk into a DataFrame."""
    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=";",
            decimal=",",
            parse_dates=["date"],
            dayfirst=True,
        )
        return df
    except Exception as e:
        print(f"  Failed to parse chunk as CSV ({e}); first 200 chars: {text[:200]!r}")
        return None



def fetch_chunk_with_retry(session, ticker, date_from, date_to, max_retries=3, pause=1.5):
    """
    Wraps fetch_chunk + parse_csv_chunk with retries and EXPLICIT failure logging.
    This is what was missing before — a chunk that failed silently (bad token,
    transient block, empty/error response) was just dropped with no trace,
    which is the likely cause of the recurring Dec-Mar gaps. Now every failure
    after all retries gets printed clearly so it's traceable, not invisible.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            text = fetch_chunk(session, ticker, date_from, date_to)
            df = parse_csv_chunk(text)
            if df is not None and len(df) > 0:
                return df
            last_error = "parsed but empty/invalid"
        except Exception as e:
            last_error = str(e)

        print(f"    retry {attempt}/{max_retries} for {ticker} "
              f"{date_from.date()}->{date_to.date()} (reason: {last_error})")
        time.sleep(pause * attempt)  # backoff a bit longer each retry

    print(f"  *** FAILED after {max_retries} retries: {ticker} "
          f"{date_from.date()}->{date_to.date()} — reason: {last_error}")
    return None



def download_full_history(ticker, start_date, end_date, chunk_days=89, pause=1.5, session=None):
    """
    Download full history for one ticker by looping over 3-month chunks,
    parsing each chunk, and concatenating into a single DataFrame.
    Returns (combined_df, failed_ranges) so callers can see exactly which
    windows failed even after retries, instead of a silent gap.
    """
    own_session = session is None
    if own_session:
        session = requests.Session()
        session.headers.update(HEADERS)

    dfs = []
    failed_ranges = []
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_days), end_date)
        print(f"  [{ticker}] {current_start.date()} -> {current_end.date()}")

        df = fetch_chunk_with_retry(session, ticker, current_start, current_end, pause=pause)
        if df is not None:
            dfs.append(df)
        else:
            failed_ranges.append((current_start, current_end))

        current_start = current_end + timedelta(days=1)
        time.sleep(pause)  # don't hammer the server

    if not dfs:
        return pd.DataFrame(), failed_ranges

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"]).sort_values("date")
    return combined, failed_ranges


if __name__ == "__main__":
    OUT_DIR = "bvmt_data"
    os.makedirs(OUT_DIR, exist_ok=True)

    start = datetime(2021, 6, 17)  # ~5 years back; trim if data quality degrades
    end = datetime(2026, 6, 17)

    session = requests.Session()
    session.headers.update(HEADERS)

    tickers = discover_tickers(session)
    if not tickers:
        print("Ticker auto-discovery failed — falling back to a manual test list.")
        tickers = [("BIAT", "BIAT")]  # add more codes here manually if needed
    else:
        print(f"Discovered {len(tickers)} tickers.")

    all_data = []
    all_failures = []
    for code, name in tickers:
        out_path = os.path.join(OUT_DIR, f"{code}.csv")
        if os.path.exists(out_path):
            print(f"Skipping {code} ({name}) — already downloaded.")
            continue

        print(f"Downloading {code} ({name})...")
        df, failed_ranges = download_full_history(code, start, end, session=session)
        if df.empty:
            print(f"  No data returned for {code}, skipping.")
            continue

        df.insert(0, "ticker_name", name)
        df.to_csv(out_path, index=False)
        all_data.append(df)
        print(f"  Saved {len(df)} rows to {out_path}")

        for f_start, f_end in failed_ranges:
            all_failures.append((code, f_start.date(), f_end.date()))

    if all_data:
        master = pd.concat(all_data, ignore_index=True)
        master.to_csv(os.path.join(OUT_DIR, "_all_tickers_combined.csv"), index=False)
        print(f"\nDone. Combined file has {len(master)} rows across {len(all_data)} tickers.")

    if all_failures:
        fail_path = os.path.join(OUT_DIR, "_failed_chunks.csv")
        pd.DataFrame(all_failures, columns=["ticker", "from", "to"]).to_csv(fail_path, index=False)
        print(f"\n{len(all_failures)} chunks failed after retries — see {fail_path} for exact gaps to re-check.")
