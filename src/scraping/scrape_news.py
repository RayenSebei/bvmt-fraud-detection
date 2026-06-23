"""
Phase 2 — Scrape per-ticker news/disclosure timelines from ilboursa.com.

Each ticker has a dedicated news archive page, e.g.:
    https://www.ilboursa.com/marches/news_valeur?s=GIF

This gives dated headlines (results announcements, court rulings, BVMT
decisions, capital increases, etc.) per company — the closest practical
proxy for "was there public information on this date" without needing
BVMT's JavaScript-heavy avis-decisions page (which couldn't be inspected
from outside due to client-side rendering).

WHY THIS MATTERS: this is what finally lets the anomaly/decay detectors
auto-classify a flagged date as EXPLAINED (matching news exists nearby —
like SOPAT's buyout, or GIF's bankruptcy ruling) vs UNEXPLAINED (no news
found at all — the genuinely suspicious case, like Tuninvest's pre-results
buying). Without this, every flag needs manual research, which doesn't scale.

Usage:
    python src/scraping/scrape_news.py
"""

import os
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.scraping.scrape_ilboursa import HEADERS

NEWS_URL = "https://www.ilboursa.com/marches/news_valeur"
OUT_DIR = "bvmt_data"
DATE_PATTERN = re.compile(r"(\d{2}/\d{2}/\d{2,4})")


def fetch_news_page(session, ticker, page=1):
    params = {"s": ticker}
    if page > 1:
        params["p"] = page  # confirmed from real pagination markup
    resp = session.get(NEWS_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.text, resp.status_code


BASE_URL = "https://www.ilboursa.com/marches/"


def parse_news_items(html):
    """
    Confirmed structure from real page source:
        <span class="sp1">12/11/24 09:34</span>
        <a href="slug-of-article_49247">Headline text</a><br />
    Date and headline are siblings — date in span.sp1, headline in the
    immediately-following <a> tag. href is a relative slug under /marches/.
    """
    soup = BeautifulSoup(html, "html.parser")
    items = []

    for span in soup.find_all("span", class_="sp1"):
        date_raw = span.get_text(strip=True)  # format: "DD/MM/YY HH:MM"
        parts = date_raw.split(" ", 1)
        date_str = parts[0] if parts else None
        time_str = parts[1] if len(parts) > 1 else None

        a = span.find_next_sibling("a")
        if a is None:
            continue

        headline = a.get_text(strip=True)
        href = a.get("href", "")
        if not href:
            continue

        full_url = href if href.startswith("http") else BASE_URL + href
        items.append({
            "date_str": date_str,
            "time_str": time_str,
            "headline": headline,
            "url": full_url,
        })

    return items


def scrape_ticker_news(session, ticker, max_pages=30, pause=1.5):
    all_items = []
    seen_urls = set()

    for page in range(1, max_pages + 1):
        print(f"  [{ticker}] fetching news page {page}...")
        try:
            html, status = fetch_news_page(session, ticker, page=page)
        except Exception as e:
            print(f"  [{ticker}] request failed on page {page}: {e}")
            break

        items = parse_news_items(html)
        new_items = [it for it in items if it["url"] not in seen_urls]

        if not new_items:
            if page == 1:
                debug_path = os.path.join(OUT_DIR, f"debug_{ticker}_page1.html")
                with open(debug_path, "w", encoding="utf-8") as f:
                    f.write(html)
                print(f"  [{ticker}] ZERO items found on page 1 (status {status}, "
                      f"{len(html)} chars). Raw HTML saved to {debug_path} for inspection.")
                print(f"  First 500 chars of response:\n{html[:500]}")
            else:
                print(f"  [{ticker}] no new items on page {page} — stopping "
                      f"(reached end of this ticker's news history).")
            break

        for it in new_items:
            seen_urls.add(it["url"])
        all_items.extend(new_items)
        time.sleep(pause)

    return all_items


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)

    test_ticker = "GIF"
    print(f"Testing news scraper on {test_ticker}...\n")
    test_items = scrape_ticker_news(session, test_ticker)

    test_df = pd.DataFrame(test_items)
    test_out = os.path.join(OUT_DIR, f"news_{test_ticker}_test.csv")
    test_df.to_csv(test_out, index=False)
    print(f"\nFound {len(test_items)} news items for {test_ticker}. Saved to {test_out}")

    if len(test_items) < 15:
        print("\nFewer items than expected from the known GIF timeline (~20).")
        print("Check the test CSV before trusting a full run — stopping here.")
    else:
        print("\nLooks right — proceeding to scrape ALL discovered tickers.")
        print("=" * 60)

        from src.scraping.scrape_ilboursa import discover_tickers
        tickers = discover_tickers(session)
        if not tickers:
            print("Ticker discovery failed — falling back to price data tickers.")
            combined_path = os.path.join(OUT_DIR, "_all_tickers_combined.csv")
            if os.path.exists(combined_path):
                existing = pd.read_csv(combined_path)
                tickers = [(t, t) for t in existing["symbole"].unique()]

        all_news = []
        for code, name in tickers:
            out_path = os.path.join(OUT_DIR, f"news_{code}.csv")
            if os.path.exists(out_path):
                print(f"Skipping {code} ({name}) — already scraped.")
                existing = pd.read_csv(out_path)
                existing.insert(0, "symbole", code)
                all_news.append(existing)
                continue

            print(f"\nScraping news for {code} ({name})...")
            items = scrape_ticker_news(session, code)
            if not items:
                print(f"  No news found for {code}, skipping.")
                continue

            df = pd.DataFrame(items)
            df.to_csv(out_path, index=False)
            df.insert(0, "symbole", code)
            all_news.append(df)
            print(f"  Saved {len(df)} items to {out_path}")

        if all_news:
            master = pd.concat(all_news, ignore_index=True)
            master_path = os.path.join(OUT_DIR, "_all_news_combined.csv")
            master.to_csv(master_path, index=False)
            print(f"\nDone. Combined news file: {len(master)} items "
                  f"across {len(all_news)} tickers -> {master_path}")
