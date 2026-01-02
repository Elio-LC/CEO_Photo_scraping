# Scraping photo Overview

Tooling to extract, validate, and score CEO photos for the "CEO Narcissism Measurement and Impact" study. The project bundles scraping, vision checks, deduping, and reporting for DEF 14A filings, and it uses supervised learning to discover repeatable patterns for locating CEO portraits. The full set of SEC DEF 14A HTML files for U.S. listed companies has already been collected.

## Core entry points (minimum set)
- `ceo_photo_pipeline_test.py`: main pipeline driver. Parses HTML, anchors candidate images, applies OpeingnCV checks, uses DashScope for verification/deduping, and writes JSON results.
- `dashscope_vision.py`: DashScope Vision API wrapper (sync/async, rate limiting).
- `progress_utils.py`: progress and timing helpers.
- `build_ceo_photo_check.py`: read CSV + photo pool, rescore, and generate an Excel QA report (can embed thumbnails).
- `check_dashscope_health.py`: DashScope availability probe.
- `html_download.py`: batch download DEF 14A HTMLs from EDGAR based on the company/year CSV; handles retries, logging, and file-lock-protected result logs.
- Common helpers: `dashscope_api_key.txt` (local API key), `execucomp_company_identifiers_annual_2000_2025_with_ceo_FINAL.csv` (base identifiers).

## Directory map
- Pipeline: `pipeline/ceo_photo_pipeline.py`, `pipeline/dashscope_vision.py`, `utils/progress_utils.py`
- Reporting/Data: `reporting/build_ceo_photo_check.py`, `reporting/export_rows_with_photos.py`, `reporting/dedupe_photos.py`, `reporting/remove_excel_images.py`
- Learning: `learning/label_ceo_photos.py`, `learning/parse_train_data.py`, `learning/ceo_photo_pattern_learner.py`, `learning/apply_learned_patterns.py`, `learning/learned_optimization.py`, `learning/feedback_learning.py`, `learning/train_and_test.py`
- Diagnostics: `diagnostics/check_dashscope_health.py`, `diagnostics/test_compare_faces.py`
- Reference docs: `COMPLETE_SYSTEM_GUIDE.txt`, `FEEDBACK_WORKFLOW.txt`, `MEMORY_SYSTEM_GUIDE.txt`, `README_PATTERN_LEARNING.md`, `QUICK_START*.txt`, `GROUP_PHOTO_HANDLING.txt`
- Data and secrets: `execucomp_company_identifiers_annual_2000_2025_with_ceo_FINAL.csv`, `dashscope_api_key.txt`, `google_api_key.txt`, `google_cse_id.txt`
- Outputs and logs (examples): `ceo_photo_results.json`, `ceo_photo_pipeline.log`, `ceo_photo_check.json`

## Installation
```bash
pip install -e .
```
Dependencies: pandas, requests, beautifulsoup4, pillow, openpyxl, opencv-python. DashScope is called directly via HTTP; put the key in `scrapephoto/dashscope_api_key.txt`.

## Quick start
1) Collect SEC DEF 14A HTMLs (if not already)  
   - Ensure `execucomp_company_identifiers_annual_2000_2025_with_ceo_FINAL.csv` is available.  
   - Run the downloader in chunks (tweak `start_tag`/`end_tag` inside the script):  
     ```bash
     python html_download.py
     ```  
   - HTMLs land under `D:/sec_data/<Company>_<CIK>/DEF_14A/...` by default; results log at `download_result.csv`.
2) Prepare keys and paths  
   - Store SEC DEF 14A HTML filings locally (e.g., `D:/sec_filings`).  
   - Place a valid DashScope key in `scrapephoto/dashscope_api_key.txt`.  
   - Default photo pool is `D:/ceo_photo_pool` (override via parameters).
3) Run the pipeline (example)  
```bash
python -m scrapephoto.pipeline.ceo_photo_pipeline \
  --csv execucomp_company_identifiers_annual_2000_2025_with_ceo_FINAL.csv \
  --sec-root D:/sec_filings \
  --shared-output D:/ceo_photo_pool \
  --results-output scrapephoto/ceo_photo_results.json \
  --log-file scrapephoto/ceo_photo_pipeline.log \
  --dashscope-concise
```
Common flags: `--limit/--offset` to bound the run; `--strategy-limit` to cap strategy count; `--brute-force-strategy` for exhaustive fallback; `--dashscope-async` to switch to async validation.
4) Generate the QA report  
```bash
python scrapephoto/reporting/build_ceo_photo_check.py \
  --csv execucomp_company_identifiers_annual_2000_2025_with_ceo_FINAL.csv \
  --photo-pool D:/ceo_photo_pool \
  --output scrapephoto/ceo_photo_check.xlsx \
  --insert-photos
```

## Helpers
- API health check: `python scrapephoto/diagnostics/check_dashscope_health.py --expect-person`

## Learning components (pattern mining)
- `label_ceo_photos.py` / `parse_train_data.py`: label samples and convert to training format.
- `ceo_photo_pattern_learner.py`: learn common location/segment patterns from labeled data.
- `apply_learned_patterns.py`: apply learned patterns to re-rank or filter candidates.
- `learned_optimization.py`: consolidate extraction outputs and emit tunable parameter summaries.
- `feedback_learning.py`: log human feedback, compute accuracy/support, and guide iterative fixes.
- `train_and_test.py`: lightweight training/evaluation harness with pluggable features and strategies.

## Outputs
- Unified photo directory: `--shared-output` (default `D:/ceo_photo_pool`)
- Results JSON: `--results-output` (default `scrapephoto/ceo_photo_results.json`)
- Logs: `--log-file` (default `scrapephoto/ceo_photo_pipeline.log`)
- Excel report: `scrapephoto/ceo_photo_check.xlsx` (optional thumbnail embedding)
