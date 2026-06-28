import subprocess
import sys
import os

def run_script(script_path):
    print(f"Running {script_path}...")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"Error running {script_path}. Exiting.")
        sys.exit(result.returncode)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    scripts = [
        "src/scraping/scrape_ilboursa.py",
        "src/scraping/scrape_news.py",
        "src/detection/anomaly_detector.py",
        "src/detection/decay_detector.py",
        "src/detection/crossref_legal_events.py",
        "src/validation/refine_watchlist.py",
        "src/triage/ai_triage_free.py",
        "src/validation/classify_anomalies.py",
        "src/validation/validate_distress_cases.py"
    ]
    
    for script in scripts:
        script_path = os.path.join(base_dir, script.replace('/', os.sep))
        run_script(script_path)
    
    print("Full pipeline completed successfully.")
