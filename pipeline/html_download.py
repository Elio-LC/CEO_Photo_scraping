#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-download DEF 14A HTML filings for many firms/years on Windows.
- Background-friendly (double-click run_bg.bat to run headless)
- Persistent error log with file lock
- Retry on failures
- Append progress to a CSV result file
"""
import os
import time
import shutil
import logging
import tempfile
import pandas as pd
from sec_edgar_downloader import Downloader
from tqdm import tqdm
import portalocker
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# Configuration
EMAIL = "acelio@ust.hk"
COMPANY = "Academic Research"
FINAL_ROOT = "D:/sec_data"
INPUT_CSV = "execucomp_company_identifiers_annual_2000_2025_with_ceo_FINAL.csv"
N_WORKERS = 4
MIN_INTERVAL = 0.5
ERROR_LOG = "down_error.log"      # Error log with advisory lock
MAX_RETRIES = 1                   # Max retry attempts
RESULT_CSV = "download_result.csv"  # Result CSV path

# Logging
logging.basicConfig(format="%(asctime)s | %(message)s",
                    level=logging.INFO,
                    handlers=[logging.StreamHandler()])
log = logging.getLogger()


def safe_name(name: str) -> str:
    """Make a filesystem-safe name by replacing invalid characters."""
    return "".join(c if c.isalnum() or c in (" ", ".", "_") else "_" for c in name).rstrip()


def check_local_exists(final_root: str, cik: str, company: str) -> bool:
    """Check whether the company's DEF 14A files already exist locally."""
    tgt_dir = os.path.join(final_root, f"{safe_name(company)}_{cik}", "DEF_14A")
    if os.path.isdir(tgt_dir) and os.listdir(tgt_dir):
        return True
    return False


def _log_error(cik: str, company: str, year: int, reason: str, attempt: int = None):
    """Write errors with file locking so multiple processes are safe."""
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        if attempt:
            f.write(f"{cik}\t{company}\t{year}\tattempt {attempt} failed\t{reason}\n")
        else:
            f.write(f"{cik}\t{company}\t{year}\t{reason}\n")
        portalocker.unlock(f)


def download_one(args):
    """Download task with retry logic for a single CIK/year."""
    cik, company, year, tmpdir, max_retries, final_root = args

    # Skip if already present
    if check_local_exists(final_root, cik, company):
        log.info(f"  {company} {year} already exists locally, skipping download")
        return cik, company, year, 1, True, True  # mark as existed

    dl = Downloader(COMPANY, EMAIL, os.path.join(tmpdir, cik))

    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            # Try downloading DEF 14A
            def_cnt = dl.get("DEF 14A", cik, after=f"{year}-01-01", before=f"{year}-12-31", download_details=True)

            # Rate limiting
            elapsed = time.time() - t0
            if elapsed < MIN_INTERVAL:
                time.sleep(MIN_INTERVAL - elapsed)

            # has_def14a: True means at least one DEF 14A was found
            has_def14a = def_cnt >= 1
            return cik, company, year, def_cnt, has_def14a, True  # task_success=True

        except Exception as exc:
            # Record error
            _log_error(cik, company, year, str(exc), attempt)

            if attempt < max_retries:
                retry_delay = attempt * 2  # exponential backoff
                log.warning(f"  {company} {year} attempt {attempt} failed, retry in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                # Out of retries
                log.error(f"  {company} {year} reached max retries {max_retries}, download failed")
                return cik, company, year, 0, False, False


def move_tmp_to_final(tmp_base: str, final_root: str, cik: str, company: str):
    """Move downloaded files from temp dir to the final destination."""
    src = os.path.join(tmp_base, cik, "sec-edgar-filings", cik)
    if not os.path.isdir(src):
        return
    tgt_top = os.path.join(final_root, f"{safe_name(company)}_{cik}")
    for form in ("DEF 14A",):
        form_src = os.path.join(src, form)
        if not os.path.isdir(form_src):
            continue
        for acc in os.listdir(form_src):
            acc_src = os.path.join(form_src, acc)
            acc_tgt = os.path.join(tgt_top, form.replace(" ", "_"), acc)
            os.makedirs(acc_tgt, exist_ok=True)
            for item in os.listdir(acc_src):
                shutil.move(os.path.join(acc_src, item), os.path.join(acc_tgt, item))
            shutil.rmtree(acc_src, ignore_errors=True)
    shutil.rmtree(src, ignore_errors=True)
    log.info(f"  Move completed -> {tgt_top}")


def init_result_csv():
    """Initialize result CSV if it does not exist."""
    if not os.path.exists(RESULT_CSV):
        columns = ["cik10", "company_name", "fyear", "def14a_count", "has_def14a", "download_status", "record_time"]
        df_init = pd.DataFrame(columns=columns)
        df_init.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
        log.info(f"Initialized result file -> {RESULT_CSV}")


def append_result_to_csv(result_data):
    """Append a single result row to CSV with locking for process safety."""
    df_row = pd.DataFrame([{
        "cik10": result_data["cik"],
        "company_name": result_data["company"],
        "fyear": result_data["year"],
        "def14a_count": result_data["def_cnt"],
        "has_def14a": result_data["has_def14a"],
        "download_status": "success" if result_data["task_success"] else "failed",
        "record_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])

    with open(RESULT_CSV, "a", encoding="utf-8-sig") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        if f.tell() > 0:
            df_row.to_csv(f, header=False, index=False)
        else:
            df_row.to_csv(f, header=True, index=False)
        portalocker.unlock(f)


def main(start_tag=201, end_tag=300, max_retries=MAX_RETRIES):
    init_result_csv()

    df = pd.read_csv(INPUT_CSV)
    df["cik"] = pd.to_numeric(df["cik"], errors='coerce')
    df = df.dropna(subset=["cik"])
    df["cik10"] = df["cik"].astype(int).astype(str).str.zfill(10)

    unique_companies = df.drop_duplicates("cik10")
    if end_tag == -1:
        end_tag = None
    target_companies = unique_companies.iloc[start_tag-1:end_tag]["cik10"].tolist()

    df_target = df[df["cik10"].isin(target_companies)].drop_duplicates(["cik10", "fyear"])
    total = len(df_target)
    log.info(f"Rows {start_tag} to {end_tag} include {total} records, "
             f"{N_WORKERS} workers, min interval {MIN_INTERVAL}s, max retries {max_retries}")
    os.makedirs(FINAL_ROOT, exist_ok=True)

    downloaded_set = set()
    if os.path.exists(RESULT_CSV):
        try:
            df_done = pd.read_csv(RESULT_CSV, encoding="utf-8-sig")
            df_done_success = df_done[
                (df_done["download_status"] == "success") &
                (df_done["has_def14a"] == True)
            ]
            downloaded_set = set(
                zip(df_done_success["cik10"].astype(str), df_done_success["fyear"].astype(int))
            )
            log.info(f"Found {len(downloaded_set)} completed records in existing log")
        except Exception as e:
            log.warning(f"Failed to read existing result log: {e}")

    df_target["task_key"] = list(zip(df_target["cik10"], df_target["fyear"].astype(int)))
    df_target_filtered = df_target[~df_target["task_key"].isin(downloaded_set)]
    total_filtered = len(df_target_filtered)
    skipped = total - total_filtered

    if skipped > 0:
        log.info(f"Skipped {skipped} already-downloaded records, {total_filtered} remaining")

    with tempfile.TemporaryDirectory() as tmpdir:
        tasks = [
            (row.cik10, row.compustat_name, int(row.fyear), tmpdir, max_retries, FINAL_ROOT)
            for _, row in df_target_filtered.iterrows()
        ]

        log.info(f"Total tasks: {len(tasks)}")

        with ProcessPoolExecutor(max_workers=N_WORKERS) as exe:
            futures = [exe.submit(download_one, t) for t in tasks]

            progress = tqdm(total=total_filtered, desc="Download progress")

            for fut in as_completed(futures):
                cik, company, year, def_cnt, has_def14a, task_success = fut.result()
                progress.update(1)

                result_data = {
                    "cik": cik,
                    "company": company,
                    "year": year,
                    "def_cnt": def_cnt,
                    "has_def14a": has_def14a,
                    "task_success": task_success
                }
                append_result_to_csv(result_data)

                if task_success:
                    log.info(f"  {company} {year}  | DEF 14A count: {def_cnt} | present: {has_def14a} (success)")
                    if has_def14a:
                        move_tmp_to_final(tmpdir, FINAL_ROOT, cik, company)
                else:
                    log.error(f"  {company} {year}  | DEF 14A count: {def_cnt} | present: {has_def14a} (failed)")

            progress.close()

    df_result = pd.read_csv(RESULT_CSV, encoding="utf-8-sig")
    total_records = len(df_result)
    success_records = len(df_result[df_result["download_status"] == "success"])
    has_def14a_count = len(df_result[df_result["has_def14a"] == True])

    log.info(f"\n==== Download summary ====")
    log.info(f"Total records: {total_records}")
    log.info(f"Successful tasks: {success_records}")
    log.info(f"Records with DEF 14A present: {has_def14a_count}")
    log.info(f"Result file: {os.path.abspath(RESULT_CSV)}")
    log.info(f"======================\n")


if __name__ == "__main__":
    # Adjust arguments to control download range and retry count
    main(start_tag=1500, end_tag=-1, max_retries=MAX_RETRIES)
