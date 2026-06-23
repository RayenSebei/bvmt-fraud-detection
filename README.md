# BVMT Market Surveillance Pipeline

[Python](https://img.shields.io/badge/python-3.10%2B-blue) [License](https://img.shields.io/badge/license-MIT-green) [Status](https://img.shields.io/badge/status-active-success)

Multi-phase market-surveillance pipeline for the Bourse de Valeurs Mobilières de Tunis (BVMT). The project scrapes historical OHLCV and news data, detects volume/price anomalies, cross-checks market-wide moves, and uses AI-assisted triage to prioritize suspicious cases for manual review. It is designed as a research workflow for studying potential insider trading, governance failures, and other event-driven distortions in an emerging-market setting.

## Table of Contents
- [Background](#background)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Usage](#usage)
- [Ground Truth and Validation](#ground-truth-and-validation)
- [Key Results](#key-results)
- [Limitations and Future Work](#limitations-and-future-work)
- [License](#license)

## Background
Emerging markets often have thinner liquidity, fewer analysts, and slower public-information diffusion than larger exchanges. That combination can make abnormal trading activity harder to interpret: some spikes are simply microstructure noise, while others may reflect meaningful information leakage, governance failures, or distress. This project explores BVMT data with a practical surveillance workflow that combines statistical anomaly detection with contextual news and market-index checks.

## Architecture
```text
scrape_ilboursa.py  ->  scrape_news.py
        |                    |
        v                    v
   anomaly_detector.py   crossref_legal_events.py
        |                    |
        v                    v
     decay_detector.py  -> classify_anomalies.py
                \            /
                 v          v
              refine_watchlist.py
                      |
                      v
                ai_triage_free.py
```

## Repository Structure
```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── docs/
├── notebooks/
├── outputs/
├── src/
│   ├── scraping/
│   ├── detection/
│   ├── triage/
│   └── validation/
├── bvmt_data/
├── *.py
├── requirements.txt
└── README.md
```

Legacy scripts currently live at the repository root for continuity, but the structure above is the recommended GitHub layout for future refactoring.

## Installation
### Conda environment
```bash
conda create -n bvmt-surveillance python=3.11
conda activate bvmt-surveillance
pip install -r requirements.txt
```

### Requirements file
If you prefer an existing environment, install the dependencies directly:
```bash
pip install -r requirements.txt
```

## Environment Setup
Sensitive credentials are loaded from a local `.env` file and are excluded from version control via `.gitignore`.

1. Copy `.env.example` to `.env`
2. Set `GROQ_API_KEY=...`
3. Keep `.env` out of Git history

The AI triage script also accepts `OPENAI_API_KEY` as a fallback, but `GROQ_API_KEY` is the preferred variable for this project.

## Usage
Run the pipeline phases in order:

1. **Scrape market data**
   ```bash
   python scrape_ilboursa.py
   ```

2. **Scrape ticker news**
   ```bash
   python scrape_news.py
   ```

3. **Detect spike anomalies**
   ```bash
   python anomaly_detector.py
   ```

4. **Detect decay/fade patterns**
   ```bash
   python decay_detector.py
   ```

5. **Cross-check legal or event context**
   ```bash
   python crossref_legal_events.py
   ```

6. **Refine the watchlist**
   ```bash
   python refine_watchlist.py
   ```

7. **Run AI-assisted triage**
   ```bash
   python ai_triage_free.py
   ```

8. **Classify and validate cases**
   ```bash
   python classify_anomalies.py
   python validate_distress_cases.py
   ```

Output files are written primarily under `bvmt_data/`.

## Ground Truth and Validation
The project was checked against documented BVMT cases and bankruptcy/delisting events, including:
- Tuninvest (TINV) insider-trading behavior flagged in Oct–Nov 2025
- UADH governance failure
- TSI Ponzi-scheme case
- CGF and UBCI documented events
- Bankruptcy-linked delistings such as GIF Filter, Electrostar, and Servicom

These cases were used to sanity-check whether the detectors surfaced plausible event windows and whether false positives were concentrated in illiquid, bursty names.

## Key Results
- The system correctly flagged the TINV Oct–Nov 2025 insider-trading anomaly.
- UADH was also validated as a meaningful signal.
- A major modeling insight was that rolling z-scores can lose sensitivity on bursty or illiquid stocks, so contextual filters and manual review remain important.

## Limitations and Future Work
- Thinly traded BVMT names can create noisy spike signals.
- Public news coverage may lag the trading event or miss the relevant catalyst entirely.
- The AI triage layer is advisory only and should not be treated as a regulator-grade conclusion.
- Future work could add alternative detectors, event-study statistics, and a more structured labeling workflow.

## License
Add your preferred open-source license here before publishing. If you want a default, MIT is a common choice for research code.