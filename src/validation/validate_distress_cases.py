"""
Phase 1.5 extension — Labeled ground-truth events + fade validation.

Two things this script does:

1. Writes bvmt_data/labeled_events.csv — combining the original documented
   cases from your fraud research (Tuninvest, UADH) with the newly
   researched delisting cases (GIF, Electrostar/LSTR, Servicom/SERVI, MIP,
   SOPAT). This becomes the reference set every detector you build from
   here on should be evaluated against, instead of ad-hoc spot checks.

2. Computes a quantitative FADE SCORE for each event: average volume in
   the 12 months immediately before the event date, divided by average
   volume in the 12 months before THAT. A score well below 1.0 means
   trading activity was genuinely fading into the event (the GIF/LSTR/
   SERVI bankruptcy pattern). A score near or above 1.0 means activity
   stayed normal or even increased right up to the event (the UADH and,
   hypothesis, SOPAT pattern — though for very different reasons: UADH
   because the market didn't know, SOPAT because the company was healthy
   and just got bought out).

This script reuses the load_data and monthly_spotlight functions from
decay_detector.py.

Usage:
    python src/validation/validate_distress_cases.py
"""

import os

import pandas as pd

from src.detection.decay_detector import load_data, monthly_spotlight

DATA_PATH = os.path.join("bvmt_data", "_all_tickers_combined.csv")
EVENTS_PATH = os.path.join("bvmt_data", "labeled_events.csv")

# Labeled ground-truth events. event_date = the date the market-facing
# action occurred (suspension date where known). category distinguishes
# the underlying mechanism, which matters for evaluating different
# detector types against different fraud/distress archetypes later.
LABELED_EVENTS = [
    {"symbole": "TINV", "ticker_name": "Tuninvest SICAR", "event_date": "2025-10-06",
     "category": "insider_trading_suspected",
     "notes": "Abnormal buying ~1 week before exceptional quarterly results; "
              "CMF lacked quorum to investigate. Caught by spike detector."},
    {"symbole": "UADH", "ticker_name": "UADH", "event_date": "2026-03-24",
     "category": "disclosure_opacity",
     "notes": "Suspended for non-publication of accounts, no AGO, no identifiable "
              "management. NO volume fade detected beforehand — invisible to decay signal."},
    {"symbole": "GIF", "ticker_name": "GIF Filter", "event_date": "2024-10-25",
     "category": "bankruptcy_distress",
     "notes": "Suspended on bankruptcy ruling (22 Oct 2024); judicial restructuring "
              "opened 2022 after unpaid wages, revenue down 85% YoY by mid-2022."},
    {"symbole": "LSTR", "ticker_name": "Electrostar", "event_date": "2024-07-23",
     "category": "bankruptcy_distress",
     "notes": "Suspended on bankruptcy ruling (payment default since July 2023); "
              "32M TND net loss in 2021, multiple cancelled capital increases."},
    {"symbole": "SERVI", "ticker_name": "Servicom", "event_date": "2024-01-11",
     "category": "bankruptcy_distress",
     "notes": "CEO left Aug 2022 with no successor (governance vacuum, echoes UADH); "
              "bankruptcy declared March 2024."},
    {"symbole": "MIP", "ticker_name": "Maghreb International Publicite", "event_date": "2024-09-10",
     "category": "unknown_cause",
     "notes": "Radiated same day as Electrostar; specific cause not found in press "
              "coverage yet — treat as exploratory, needs more research."},
    {"symbole": "SOPAT", "ticker_name": "SOPAT", "event_date": "2023-09-20",
     "category": "benign_delisting_buyout",
     "notes": "NEGATIVE CONTROL. Majority shareholder (>95%) launched buyout offer "
              "while company was profitable (+7x profit in 2023). NOT distress — "
              "tests whether the detector can be fooled by a benign delisting."},
]


def save_labeled_events():
    df = pd.DataFrame(LABELED_EVENTS)
    df.to_csv(EVENTS_PATH, index=False)
    print(f"Saved {len(df)} labeled events to {EVENTS_PATH}")
    return df


def fade_score(df, symbole, event_date, months_far=24, months_near=12):
    """
    near_window = [event - 12mo, event)
    far_window  = [event - 24mo, event - 12mo)
    score = avg(near_volume) / avg(far_volume)

    score << 1.0  -> activity was fading heading into the event
    score ~= 1.0  -> no fade, activity stayed normal
    score > 1.0   -> activity actually increased heading into the event
    """
    g = df[df["symbole"] == symbole].copy()
    if g.empty:
        return None

    event = pd.Timestamp(event_date)
    far_start = event - pd.DateOffset(months=months_far)
    near_start = event - pd.DateOffset(months=months_near)

    far = g[(g["date"] >= far_start) & (g["date"] < near_start)]
    near = g[(g["date"] >= near_start) & (g["date"] < event)]

    far_avg = far["volume"].mean() if len(far) else None
    near_avg = near["volume"].mean() if len(near) else None

    if not far_avg or far_avg == 0 or near_avg is None:
        return None

    return near_avg / far_avg


if __name__ == "__main__":
    print("Saving labeled ground-truth events...")
    events_df = save_labeled_events()

    print("\nLoading price/volume data...")
    df = load_data(DATA_PATH)

    print("\n" + "=" * 70)
    print("FADE SCORE — recent-12mo avg volume vs prior-12mo avg volume,")
    print("measured up to each event date.")
    print("=" * 70)

    for ev in LABELED_EVENTS:
        score = fade_score(df, ev["symbole"], ev["event_date"])
        score_str = f"{score:.3f}" if score is not None else "N/A (insufficient data)"
        print(f"\n{ev['symbole']} ({ev['ticker_name']}) — {ev['category']}")
        print(f"  Event date: {ev['event_date']}  |  Fade score: {score_str}")

    print("\n" + "=" * 70)
    print("MONTHLY SPOTLIGHTS — bankruptcy cases vs the benign control")
    print("=" * 70)

    for symbole in ["GIF", "LSTR", "SERVI", "SOPAT"]:
        print(f"\n--- {symbole} ---")
        monthly_spotlight(df, symbole, months=24)

    print("\n" + "=" * 70)
    print("WHAT TO LOOK FOR")
    print("=" * 70)
    print("- GIF, LSTR, SERVI (bankruptcy): expect fade scores well below 1.0 and")
    print("  visibly declining trading_days/volume in the months before suspension.")
    print("- SOPAT (benign buyout, negative control): expect a fade score closer")
    print("  to 1.0, or volume that looks healthy right up to the event — since")
    print("  the company itself was NOT in distress, just acquired.")
    print("- If SOPAT's score looks just as low as the bankruptcy cases, that's an")
    print("  important finding: it means volume decline ALONE can't distinguish")
    print("  'distress' from 'benign delisting' — you'd need to cross-reference")
    print("  BVMT's avis-decisions bulletin (Phase 2) to tell them apart reliably.")
    print("- MIP: no strong hypothesis yet, treat as exploratory.")
