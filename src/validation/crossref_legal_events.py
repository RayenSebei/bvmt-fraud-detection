"""
Cross-reference the Phase 1 spike detector's flags against known legal/court
event dates for GIF and Electrostar (LSTR), researched from press coverage.
Tests the hypothesis that their bankruptcy processes showed up as discrete
volume/price BURSTS around specific news dates, rather than a smooth fade
(which the decay detector already showed they did NOT have).

Usage:
    python crossref_legal_events.py
Reads bvmt_data/anomaly_flags.csv (from anomaly_detector.py — run that first
if you haven't already).
"""

import os

import pandas as pd

FLAGS_PATH = os.path.join("bvmt_data", "anomaly_flags.csv")
WINDOW_DAYS = 5  # how close a flagged date must be to a known event to count as a match

# Known legal/court dates researched from press coverage, in addition to the
# final suspension date already in labeled_events.csv. These are the moments
# the market would plausibly have reacted to, if it reacted at all.
KNOWN_EVENTS = [
    ("GIF", "2022-10-17", "Ouverture procedure redressement judiciaire"),
    ("GIF", "2023-07-06", "Justice decide mise en vente"),
    ("GIF", "2023-08-28", "Tribunal reporte la vente judiciaire"),
    ("GIF", "2024-10-22", "Tribunal prononce la faillite"),
    ("GIF", "2024-10-25", "Suspension de cotation"),
    ("GIF", "2024-11-07", "Radiation decidee"),

    ("LSTR", "2023-10-25", "Ouverture procedure redressement judiciaire"),
    ("LSTR", "2024-02-05", "Annonce augmentation de capital 18M TND"),
    ("LSTR", "2024-02-29", "Report augmentation de capital"),
    ("LSTR", "2024-04-18", "Annulation augmentation de capital"),
    ("LSTR", "2024-07-15", "Jugement tribunal - faillite declaree"),
    ("LSTR", "2024-07-23", "Suspension de cotation"),
    ("LSTR", "2024-09-05", "Radiation decidee"),
]


def load_flags(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def check_matches(flags, symbole, event_date, label):
    event = pd.Timestamp(event_date)
    g = flags[flags["symbole"] == symbole].copy()
    g["days_from_event"] = (g["date"] - event).dt.days

    nearby = g[g["days_from_event"].abs() <= WINDOW_DAYS]
    return nearby


if __name__ == "__main__":
    print("Loading anomaly flags...")
    flags = load_flags(FLAGS_PATH)

    print(f"\nChecking for flagged anomalies within +/-{WINDOW_DAYS} days of known legal events...\n")
    print("=" * 80)

    total_matches = 0
    for symbole, event_date, label in KNOWN_EVENTS:
        nearby = check_matches(flags, symbole, event_date, label)
        status = f"{len(nearby)} MATCH(ES)" if len(nearby) > 0 else "no match"
        print(f"\n{symbole} | {event_date} | {label}")
        print(f"  -> {status}")
        if len(nearby) > 0:
            total_matches += len(nearby)
            cols = ["date", "volume_zscore", "return_zscore", "combined_anomaly", "days_from_event"]
            print(nearby[cols].to_string(index=False))

    print("\n" + "=" * 80)
    print(f"TOTAL: {total_matches} flagged anomalies fell within {WINDOW_DAYS} days of a known legal event.")
    print("=" * 80)
    print("\nINTERPRETATION:")
    print("- If most known events have a nearby match: the spike detector IS catching")
    print("  these legal/court dates as discrete bursts, even though the decay detector")
    print("  found no smooth fade. That reframes GIF/LSTR as 'spike' cases, not 'decay'.")
    print("- If few/no matches: neither detector type caught these specific events,")
    print("  meaning the market reaction (if any) didn't show up as a 3-sigma anomaly,")
    print("  or happened on a different day than the news (settlement lag, thin trading).")
