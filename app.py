"""
BVMT Market Surveillance Dashboard - Backend API
Reads real CSV data from your bvmt_data folder.
No fake data. All endpoints serve real scraped files.
"""

import os
import json
import math
import threading
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask.json.provider import DefaultJSONProvider


class NaNSafeJSONProvider(DefaultJSONProvider):
    """Custom JSON provider that converts NaN/Inf floats to None (JSON null)."""
    def default(self, o):
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        return super().default(o)

    def dumps(self, obj, **kwargs):
        return super().dumps(obj, **kwargs)


def safe_val(v, fallback=None):
    """Return fallback if v is NaN/None/NaT, else v."""
    try:
        if v is None:
            return fallback
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return fallback
        if pd.isna(v):
            return fallback
    except (TypeError, ValueError):
        pass
    return v

app = Flask(__name__)
app.json_provider_class = NaNSafeJSONProvider
app.json = NaNSafeJSONProvider(app)
CORS(app)

# ── DATA DIRECTORY ──────────────────────────────────────────────────────────
# On Render: set DATA_DIR env var to your mounted data path
# Locally:   set DATA_DIR or defaults to ./data
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))

# ── SCRAPER SCRIPT PATH ──────────────────────────────────────────────────────
# Point this to your existing scraper/pipeline entry point
SCRAPER_SCRIPT = os.environ.get("SCRAPER_SCRIPT", "./scraper/run_pipeline.py")

# ── SCRAPE STATE ─────────────────────────────────────────────────────────────
scrape_state = {
    "running": False,
    "last_run": None,
    "last_status": "never",
    "log": []
}


def load_csv(filename: str, **kwargs) -> pd.DataFrame:
    """Load a CSV from DATA_DIR. Returns empty DataFrame on error."""
    path = DATA_DIR / filename
    if not path.exists():
        app.logger.warning(f"File not found: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        app.logger.error(f"Error reading {path}: {e}")
        return pd.DataFrame()


def run_scraper_background():
    """Run the pipeline scraper in a background thread."""
    global scrape_state
    scrape_state["running"] = True
    scrape_state["log"] = []
    scrape_state["last_status"] = "running"
    try:
        result = subprocess.run(
            ["python", SCRAPER_SCRIPT],
            capture_output=True, text=True, timeout=300
        )
        scrape_state["log"] = result.stdout.splitlines()[-30:]
        scrape_state["last_status"] = "success" if result.returncode == 0 else "error"
    except FileNotFoundError:
        scrape_state["last_status"] = "error"
        scrape_state["log"] = ["Scraper script not found at: " + SCRAPER_SCRIPT]
    except subprocess.TimeoutExpired:
        scrape_state["last_status"] = "timeout"
        scrape_state["log"] = ["Scraper timed out after 5 minutes"]
    except Exception as e:
        scrape_state["last_status"] = "error"
        scrape_state["log"] = [str(e)]
    finally:
        scrape_state["running"] = False
        scrape_state["last_run"] = datetime.now().isoformat()


# ── API ROUTES ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/status")
def api_status():
    """Pipeline health + last scrape info."""
    files = {
        "tickers": (DATA_DIR / "_all_tickers_combined.csv").exists(),
        "news": (DATA_DIR / "_all_news_combined.csv").exists(),
        "anomaly_flags": (DATA_DIR / "anomaly_flags.csv").exists(),
        "anomaly_classified": (DATA_DIR / "anomaly_classified.csv").exists(),
        "anomaly_summary": (DATA_DIR / "anomaly_summary.csv").exists(),
    }
    return jsonify({
        "files": files,
        "scraper": scrape_state,
        "data_dir": str(DATA_DIR)
    })


@app.route("/api/scrape", methods=["POST"])
def api_scrape():
    """Trigger a background scrape. Called when dashboard loads."""
    if scrape_state["running"]:
        return jsonify({"message": "Scraper already running", "state": scrape_state})
    t = threading.Thread(target=run_scraper_background, daemon=True)
    t.start()
    return jsonify({"message": "Scraper started", "state": scrape_state})


@app.route("/api/scrape/status")
def api_scrape_status():
    return jsonify(scrape_state)


@app.route("/api/kpis")
def api_kpis():
    """Summary KPIs for the top bar."""
    tickers_df = load_csv("_all_tickers_combined.csv")
    news_df    = load_csv("_all_news_combined.csv")
    flags_df   = load_csv("anomaly_flags.csv")
    classified = load_csv("anomaly_classified.csv")

    n_tickers   = tickers_df["symbole"].nunique() if not tickers_df.empty else 0
    n_news      = len(news_df) if not news_df.empty else 0

    # Flagged = combined_anomaly == True in last 30 days
    flagged_30d = 0
    if not flags_df.empty and "date" in flags_df.columns:
        flags_df["date"] = pd.to_datetime(flags_df["date"], errors="coerce")
        cutoff = datetime.now() - timedelta(days=30)
        recent = flags_df[flags_df["date"] >= cutoff]
        if "combined_anomaly" in recent.columns:
            col = recent["combined_anomaly"].astype(str).str.strip().str.lower()
            vol_col = recent["volume_anomaly"].astype(str).str.strip().str.lower() if "volume_anomaly" in recent.columns else pd.Series(dtype=str)
            pri_col = recent["price_anomaly"].astype(str).str.strip().str.lower() if "price_anomaly" in recent.columns else pd.Series(dtype=str)
            flagged_30d = int(((col == "true") | (vol_col == "true") | (pri_col == "true")).sum())

    # EXPLAINED count
    explained = 0
    if not classified.empty and "status" in classified.columns:
        explained = int((classified["status"] == "EXPLAINED").sum())

    total_classified = len(classified) if not classified.empty else 0
    triage_rate = round(explained / total_classified * 100, 1) if total_classified else 0

    return jsonify({
        "n_tickers": n_tickers,
        "n_news": n_news,
        "flagged_30d": flagged_30d,
        "triage_rate": triage_rate,
        "total_classified": total_classified,
        "explained": explained
    })


@app.route("/api/anomalies")
def api_anomalies():
    """Top anomalies sorted by highest absolute z-score."""
    df = load_csv("anomaly_classified.csv")
    if df.empty:
        return jsonify([])

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")

    # Keep only actual anomalies (volume or price flagged)
    mask = df["volume_anomaly"].astype(str).str.strip().str.lower() == "true"
    mask = mask | (df["price_anomaly"].astype(str).str.strip().str.lower() == "true")
    df = df[mask].copy()

    # Highest absolute z-score wins
    df["max_zscore"] = df[["volume_zscore", "return_zscore"]].abs().max(axis=1)
    df = df.sort_values("max_zscore", ascending=False).head(50)

    # Determine dominant signal
    def signal_type(row):
        va = str(row.get("volume_anomaly", "")).strip().lower() == "true"
        pa = str(row.get("price_anomaly", "")).strip().lower() == "true"
        if va and pa:
            return "Volume + Price"
        elif va:
            return "Volume spike"
        elif pa:
            return "Price anomaly"
        return "Unknown"

    def risk_level(z):
        if abs(z) >= 4.0:  return "High"
        elif abs(z) >= 2.5: return "Medium"
        return "Low"

    records = []
    for _, row in df.iterrows():
        records.append({
            "symbole":          safe_val(row.get("symbole"), ""),
            "ticker_name":      safe_val(row.get("ticker_name"), ""),
            "date":             str(row["date"].date()) if pd.notna(row["date"]) else "",
            "cloture":          round(float(row["cloture"]), 3) if pd.notna(row.get("cloture")) else None,
            "volume":           int(row["volume"]) if pd.notna(row.get("volume")) else None,
            "volume_zscore":    round(float(row["volume_zscore"]), 2) if pd.notna(row.get("volume_zscore")) else None,
            "return_zscore":    round(float(row["return_zscore"]), 2) if pd.notna(row.get("return_zscore")) else None,
            "max_zscore":       round(float(row["max_zscore"]), 2) if pd.notna(row.get("max_zscore")) else None,
            "signal":           signal_type(row),
            "risk":             risk_level(row["max_zscore"]) if pd.notna(row.get("max_zscore")) else "Unknown",
            "status":           safe_val(row.get("status"), ""),
            "matched_headline": safe_val(row.get("matched_headline"), ""),
            "category":         safe_val(row.get("category"), ""),
        })
    return jsonify(records)



@app.route("/api/anomalies/timeline")
def api_anomaly_timeline():
    """Daily anomaly count for the past 90 days — for chart."""
    df = load_csv("anomaly_flags.csv")
    if df.empty:
        return jsonify([])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    cutoff = datetime.now() - timedelta(days=90)
    df = df[df["date"] >= cutoff]

    if "combined_anomaly" not in df.columns:
        return jsonify([])

    ca = df["volume_anomaly"].astype(str).str.strip().str.lower() == "true"
    ca = ca | (df["price_anomaly"].astype(str).str.strip().str.lower() == "true")
    daily = (
        df[ca]
        .groupby(df["date"].dt.date)
        .size()
        .reset_index(name="count")
    )
    daily["date"] = daily["date"].astype(str)
    return jsonify(daily.to_dict(orient="records"))


@app.route("/api/news/recent")
def api_news_recent():
    """Most recent 30 news items."""
    df = load_csv("_all_news_combined.csv")
    if df.empty:
        return jsonify([])

    # Parse date from date_str column (format: DD/MM/YY)
    df["parsed_date"] = pd.to_datetime(df["date_str"], format="%d/%m/%y", errors="coerce")
    df = df.sort_values("parsed_date", ascending=False).head(30)

    records = []
    for _, row in df.iterrows():
        records.append({
            "symbole":   row.get("symbole", ""),
            "date":      row.get("date_str", ""),
            "time":      row.get("time_str", ""),
            "headline":  row.get("headline", ""),
            "url":       row.get("url", ""),
        })
    return jsonify(records)


@app.route("/api/news/by_ticker")
def api_news_by_ticker():
    """News for a specific ticker. ?ticker=BIAT"""
    ticker = request.args.get("ticker", "").upper()
    if not ticker:
        return jsonify([])

    df = load_csv("_all_news_combined.csv")
    if df.empty:
        return jsonify([])

    df = df[df["symbole"].str.upper() == ticker]
    df["parsed_date"] = pd.to_datetime(df["date_str"], format="%d/%m/%y", errors="coerce")
    df = df.sort_values("parsed_date", ascending=False)

    records = []
    for _, row in df.iterrows():
        records.append({
            "symbole":  row.get("symbole", ""),
            "date":     row.get("date_str", ""),
            "time":     row.get("time_str", ""),
            "headline": row.get("headline", ""),
            "url":      row.get("url", ""),
        })
    return jsonify(records)


@app.route("/api/stock/<ticker>")
def api_stock(ticker):
    """OHLCV history for a ticker. Returns last 252 trading days."""
    ticker = ticker.upper()
    df = load_csv(f"{ticker}.csv")

    if df.empty:
        # Try combined file
        combined = load_csv("_all_tickers_combined.csv")
        if not combined.empty and "symbole" in combined.columns:
            df = combined[combined["symbole"].str.upper() == ticker].copy()

    if df.empty:
        return jsonify({"error": f"No data found for {ticker}"}), 404

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").tail(252)

    # Get anomaly flags for this ticker
    flags = load_csv("anomaly_classified.csv")
    anomaly_dates = set()
    if not flags.empty and "symbole" in flags.columns:
        tf = flags[flags["symbole"].str.upper() == ticker]
        if not tf.empty:
            tf["date"] = pd.to_datetime(tf["date"], errors="coerce")
            va = tf["volume_anomaly"].astype(str).str.strip().str.lower() == "true"
            pa = tf["price_anomaly"].astype(str).str.strip().str.lower() == "true"
            anomaly_dates = set(tf[va | pa]["date"].dt.strftime("%Y-%m-%d"))

    records = []
    for _, row in df.iterrows():
        d = str(row["date"].date()) if pd.notna(row["date"]) else ""
        records.append({
            "date":      d,
            "ouverture": round(float(row["ouverture"]), 3) if pd.notna(row.get("ouverture")) else None,
            "haut":      round(float(row["haut"]), 3)      if pd.notna(row.get("haut"))      else None,
            "bas":       round(float(row["bas"]), 3)        if pd.notna(row.get("bas"))       else None,
            "cloture":   round(float(row["cloture"]), 3)   if pd.notna(row.get("cloture"))   else None,
            "volume":    int(row["volume"])                 if pd.notna(row.get("volume"))    else None,
            "anomaly":   d in anomaly_dates,
        })
    return jsonify(records)


@app.route("/api/tickers")
def api_tickers():
    """List of all available ticker symbols."""
    combined = load_csv("_all_tickers_combined.csv")
    if not combined.empty and "symbole" in combined.columns:
        tickers = sorted(combined["symbole"].dropna().unique().tolist())
        names = {}
        if "ticker_name" in combined.columns:
            names = combined.drop_duplicates("symbole").set_index("symbole")["ticker_name"].to_dict()
        return jsonify([{"symbole": t, "name": names.get(t, t)} for t in tickers])

    # Fallback: scan individual CSV files
    csvs = [f.stem for f in DATA_DIR.glob("*.csv")
            if not f.stem.startswith("_") and not f.stem.startswith("anomaly")]
    return jsonify([{"symbole": t, "name": t} for t in sorted(csvs)])


@app.route("/api/summary")
def api_summary():
    """anomaly_summary.csv contents."""
    df = load_csv("anomaly_summary.csv")
    if df.empty:
        return jsonify([])
    return jsonify(df.fillna("").to_dict(orient="records"))


@app.route("/api/news/categories")
def api_news_categories():
    """Count news items per category from anomaly_classified."""
    df = load_csv("anomaly_classified.csv")
    if df.empty or "category" not in df.columns:
        return jsonify([])
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")

    cats = df["category"].dropna()
    cats = cats[cats != ""]
    counts = cats.value_counts().reset_index()
    counts.columns = ["category", "count"]
    return jsonify(counts.to_dict(orient="records"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
