"""
scraper/run_pipeline.py
=======================
This file is called by the dashboard backend whenever a user opens the page.
It runs in the background (non-blocking) and updates the CSV files in ./data/

EDIT THIS FILE to point to your actual pipeline entry point.
"""

import subprocess
import sys
import os

# ── OPTION A: Call your existing main scraper script ────────────────────────
# Change this path to wherever your pipeline entry point is
PIPELINE_SCRIPT = os.environ.get(
    "PIPELINE_SCRIPT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "run_all.py")),
)

if __name__ == "__main__":
    print(f"[Scraper] Starting pipeline: {PIPELINE_SCRIPT}")
    result = subprocess.run(
        [sys.executable, PIPELINE_SCRIPT],
        capture_output=False,
        text=True
    )
    if result.returncode == 0:
        print("[Scraper] Pipeline completed successfully.")
    else:
        print(f"[Scraper] Pipeline exited with code {result.returncode}")
        sys.exit(result.returncode)
