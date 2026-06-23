"""
AI-assisted triage layer for the BVMT market surveillance watchlist —
Using Groq's free API (completely free, 30 requests/minute, no credit card required).

This does NOT replace the quantitative pipeline (spike/decay detectors,
news matching, index cross-check) — it's an additional layer that reads
the structured evidence you've already computed and writes a short,
human-readable judgement to help you prioritize manual research.

IMPORTANT: treat its output as a draft opinion to assist YOUR review, not
a verdict. The model has no information beyond the numbers you give it,
can be wrong, and should never be quoted as "AI confirmed this is
suspicious" in any writeup. Frame it as "AI-assisted triage suggested X,
pending manual verification."

Groq Free Tier Limits:
    - 30 requests per minute
    - Completely free, no credit card needed
    - Models available: llama-3.3-70b, mixtral-8x7b, gemma2-9b

Reads:
    bvmt_data/watchlist_refined.csv
Writes:
    bvmt_data/watchlist_ai_assessed.csv

Usage:
    python ai_triage_free.py
"""

import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")

if not GROQ_API_KEY:
    raise SystemExit(
        "Missing API key. Set GROQ_API_KEY in your .env file (or OPENAI_API_KEY as a fallback)."
    )

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

MODEL = "llama-3.3-70b-versatile"
OUT_DIR = "bvmt_data"
IN_PATH = os.path.join(OUT_DIR, "watchlist_refined.csv")
OUT_PATH = os.path.join(OUT_DIR, "watchlist_ai_assessed.csv")

PAUSE_SECONDS = 3  # 30 req/min -> 2 seconds between calls, padded to 3 for safety

SYSTEM_PROMPT = """You are assisting a student building a market-surveillance \
research tool for the Tunis Stock Exchange (BVMT). You will be given the \
quantitative signal behind one flagged trading anomaly (volume/price \
z-scores, or a liquidity-decline ratio) for one stock, plus whether any \
matching company news was found nearby and whether the move coincided \
with a broad market move.

Your job is ONLY to assess plausibility based on the numbers given. You \
have no information beyond what's provided, you are not a financial \
regulator, and you cannot confirm fraud. Respond with STRICT JSON only, \
no other text, in this exact shape:
{"assessment": "likely_noise" | "worth_investigating" | "uncertain",
 "reasoning": "<one or two sentences, plain language, citing the specific \
numbers that drove your judgement>"}

Guidance:
- Very high z-scores (>5) on otherwise illiquid/rarely-flagged stocks are \
often just a side effect of thin trading, not necessarily meaningful.
- A pattern of REPEATED flags for the same ticker over years suggests \
chronic illiquidity noise, not a one-off event.
- Genuinely worth flagging: large, isolated moves (high excess_return, \
low co_flagged_count) on tickers that are NOT chronic offenders, with no \
news and no market-wide explanation.
- Be conservative. When uncertain, say so — do not invent explanations \
or make accusations.
"""

def build_user_prompt(row, repeat_count):
    fields = {
        "ticker": row.get("symbole"),
        "company_name": row.get("ticker_name"),
        "date": str(row.get("date")),
        "source_detector": row.get("source"),
        "volume_zscore": row.get("volume_zscore"),
        "return_zscore": row.get("return_zscore"),
        "decline_ratio": row.get("decline_ratio"),
        "excess_return_vs_index": row.get("excess_return"),
        "co_flagged_other_tickers_same_day": row.get("co_flagged_count"),
        "tag_from_index_check": row.get("tag"),
        "times_this_ticker_appears_in_watchlist": repeat_count,
    }
    return "Assess this flagged anomaly:\n" + json.dumps(fields, default=str, indent=2)

def assess_row(row, repeat_count, max_retries=3):
    user_prompt = build_user_prompt(row, repeat_count)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = resp.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)
            return parsed.get("assessment"), parsed.get("reasoning")
        except Exception as e:
            wait = 20 * (attempt + 1) if "429" in str(e) else 2 * (attempt + 1)
            print(f"  retry {attempt + 1}/{max_retries} for {row.get('symbole')}: {e} "
                  f"(waiting {wait}s)")
            time.sleep(wait)

    return "error", "Failed to get a valid response after retries."

if __name__ == "__main__":
    print("=" * 70)
    print("AI TRIAGE WITH GROQ FREE API")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Rate Limit: 30 requests/minute (waiting {PAUSE_SECONDS}s between calls)")
    print("=" * 70)
    
    if not os.path.exists(IN_PATH):
        print(f"ERROR: Input file not found: {IN_PATH}")
        print("Make sure you've run the main pipeline first to generate watchlist_refined.csv")
        raise SystemExit(1)

    df = pd.read_csv(IN_PATH)
    print(f"\nLoaded {len(df)} watchlist rows.")
    print(f"This will take roughly {len(df) * PAUSE_SECONDS / 60:.0f} minutes.\n")

    repeat_counts = df["symbole"].value_counts()

    assessments, reasonings = [], []
    for i, row in df.iterrows():
        symbole = row.get("symbole")
        print(f"[{i + 1}/{len(df)}] Assessing {symbole}...")
        assessment, reasoning = assess_row(row, int(repeat_counts.get(symbole, 1)))
        assessments.append(assessment)
        reasonings.append(reasoning)
        time.sleep(PAUSE_SECONDS)

    df["ai_assessment"] = assessments
    df["ai_reasoning"] = reasonings
    
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(df["ai_assessment"].value_counts().to_string())
    print(f"\nSaved: {OUT_PATH}")

    print("\nTop 'worth_investigating' rows:")
    top = df[df["ai_assessment"] == "worth_investigating"]
    if len(top) > 0:
        cols = [c for c in ["symbole", "ticker_name", "date", "ai_reasoning"] if c in top.columns]
        print(top[cols].to_string(index=False))
    else:
        print("  None found.")

    print("\n" + "=" * 70)
    print("REMINDER: these are AI-assisted draft judgements based only on the")
    print("numbers provided, not verified conclusions. Manually review before")
    print("citing any of these as findings.")
    print("=" * 70)