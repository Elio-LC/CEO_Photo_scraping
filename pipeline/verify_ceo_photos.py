import argparse
import csv
import logging
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from dashscope_vision import DashscopeVisionClient

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CEO_NAME_COLUMNS = [
    "ceo_name",
    "ceo_full_name",
    "ceo_fullname",
    "ceo",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify CEO photos across years")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("scrapephoto") / "execucomp_company_identifiers_annual_2000_2025_with_ceo_FINAL.csv",
        help="Path to execucomp CSV dataset.",
    )
    parser.add_argument(
        "--sec-data",
        type=Path,
        default=Path("D:/ceo_photo_pool"),
        help="Root directory containing sec_data/<company>_<cik>/ceophoto folders (default: D:/ceo_photo_pool).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("verify_ceo_photos_report.csv"),
        help="Output CSV report path.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("verify_ceo_photos.log"),
        help="Log file path.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path("D:/delete"),
        help="Directory where flagged photos will be moved (default: D:/delete).",
    )
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=0.95,
        help="Confidence threshold to treat verdict=='different' as actionable.",
    )
    return parser.parse_args()


def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def sanitize_company_name(name: str) -> str:
    clean = "".join(ch if ch.isalnum() else "_" for ch in name.strip())
    return "_".join(filter(None, clean.split("_"))).lower()


def build_ceophoto_dir(sec_root: Path, company: str, cik10: str) -> Path:
    safe_name = sanitize_company_name(company or "unknown")
    return sec_root / f"{safe_name}_{cik10}" / "ceophoto"


def extract_ceo_name(row: pd.Series) -> str:
    for column in CEO_NAME_COLUMNS:
        if column not in row:
            continue
        value = row.get(column)
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def index_company_photos(directory: Path) -> Dict[int, List[Path]]:
    mapping: Dict[int, List[Path]] = defaultdict(list)
    if not directory.exists():
        return mapping
    for entry in directory.iterdir():
        if not entry.is_file() or entry.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = entry.stem
        digits = []
        for ch in stem:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if not digits:
            continue
        try:
            key = int("".join(digits))
        except ValueError:
            continue
        mapping[key].append(entry)
    return mapping


def _index_pool_files_for_cik(sec_root: Path, cik10: str) -> Dict[int, List[Path]]:
    mapping: Dict[int, List[Path]] = defaultdict(list)
    pattern = f"{cik10}_*"
    for path in sec_root.glob(pattern):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        remainder = path.stem[len(cik10) + 1 :]
        digits = []
        for ch in remainder:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if not digits:
            continue
        digit_str = "".join(digits)
        if len(digit_str) >= 4:
            digit_str = digit_str[:4]
        try:
            year = int(digit_str)
        except ValueError:
            continue
        mapping[year].append(path)
    return mapping


def find_photo_for_year(
    year: int,
    cik10: str,
    directory: Path,
    sec_root: Path,
    dir_cache: Dict[Path, Dict[int, List[Path]]],
    pool_cache: Dict[str, Dict[int, List[Path]]],
) -> Optional[Path]:
    if directory not in dir_cache:
        dir_cache[directory] = index_company_photos(directory)
    mapping = dir_cache[directory]
    candidates: List[Path] = []
    keys = {year, year % 100}
    for key in keys:
        candidates.extend(mapping.get(key, []))
    if candidates:
        return sorted(candidates)[0]

    if cik10 not in pool_cache:
        pool_cache[cik10] = _index_pool_files_for_cik(sec_root, cik10)
    pool_mapping = pool_cache[cik10]
    pool_candidates: List[Path] = []
    for key in keys:
        pool_candidates.extend(pool_mapping.get(key, []))
    if pool_candidates:
        return sorted(pool_candidates)[0]
    return None


def move_with_structure(source: Path, sec_root: Path, backup_root: Path) -> Path:
    try:
        rel = source.relative_to(sec_root)
    except ValueError:
        rel = source.name
    target = backup_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    return target


def group_records(df: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, Any]]:
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for _, row in df.iterrows():
        ceo_name = extract_ceo_name(row)
        if not ceo_name:
            continue
        cik_val = row.get("cik")
        try:
            cik10 = f"{int(cik_val):010d}"
        except (TypeError, ValueError):
            continue
        company = str(row.get("compustat_name") or row.get("primary_company_name") or "").strip()
        if not company:
            continue
        fyear = row.get("fyear")
        try:
            year_int = int(fyear)
        except (TypeError, ValueError):
            continue
        key = (cik10, ceo_name.lower())
        entry = groups.setdefault(
            key,
            {
                "cik10": cik10,
                "ceo_name": ceo_name,
                "company": company,
                "records": [],
            },
        )
        entry["records"].append(year_int)
    return groups


def write_report(report_path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        logging.info("No rows to write to report %s", report_path)
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "company",
        "cik",
        "ceo_name",
        "years",
        "verdict",
        "confidence",
        "majority_indices",
        "majority_years",
        "outlier_indices",
        "outlier_years",
        "multi_indices",
        "multi_years",
        "fallback_used",
        "to_delete",
        "action",
        "timestamp",
    ]
    with report_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_file)
    sec_root = args.sec_data.resolve()
    backup_root = args.backup_dir.resolve()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    df = pd.read_csv(args.csv)
    groups = group_records(df)
    logging.info("Loaded %d CEO groups from %s", len(groups), args.csv)
    if not groups:
        logging.warning(
            "No CEO groups detected; ensure CSV has columns: %s",
            ", ".join(CEO_NAME_COLUMNS),
        )

    client = DashscopeVisionClient()
    if not client.is_ready():
        logging.error("DashScope client not ready; please configure API key.")
        return

    photo_cache: Dict[Path, Dict[int, List[Path]]] = {}
    pool_photo_cache: Dict[str, Dict[int, List[Path]]] = {}
    report_rows: List[Dict[str, str]] = []

    for (cik10, ceo_key), entry in sorted(groups.items()):
        ceo_name = entry["ceo_name"]
        company = entry["company"]
        ceo_dir = build_ceophoto_dir(sec_root, company, cik10)
        available: List[Tuple[int, Path]] = []
        for year in sorted(set(entry["records"])):
            photo_path = find_photo_for_year(
                year,
                cik10,
                ceo_dir,
                sec_root,
                photo_cache,
                pool_photo_cache,
            )
            if photo_path is None:
                continue
            available.append((year, photo_path))
        if len(available) < 2:
            continue
        available.sort(key=lambda item: item[0])
        image_paths = [str(path) for _, path in available]
        index_map = {idx + 1: entry for idx, entry in enumerate(available)}
        verdict, confidence, majority_idx, outlier_idx, multi_idx = client.compare_faces_detailed(image_paths)
        logging.info(
            "[%s|%s|%s] verdict=%s confidence=%.3f majority=%s outliers=%s multi=%s",
            company,
            cik10,
            ceo_name,
            verdict,
            confidence,
            majority_idx,
            outlier_idx,
            multi_idx,
        )

        to_delete: List[Path] = []
        action = "kept"
        moved_paths: List[str] = []

        canonical_idx = [idx for idx in majority_idx if idx in index_map]
        fallback_used = False
        multi_detected = bool(multi_idx)
        majority_confident = len(canonical_idx) >= 2
        if not majority_confident and verdict not in {"multi"} and not multi_detected:
            fallback_used = True
            latest_idx = max(index_map.keys())
            canonical_idx = [latest_idx]

        if verdict == "multi" or multi_detected:
            action = "multi_person_auto_kept"
            fallback_used = False
            logging.info(
                "Multi-face photo detected for %s (%s); group kept without deletion.",
                ceo_name,
                company,
            )
        elif majority_confident:
            removable_idx = [idx for idx in index_map if idx not in canonical_idx and idx not in multi_idx]
            to_delete = [index_map[idx][1] for idx in removable_idx]
            for path in to_delete:
                moved_path = move_with_structure(Path(path), sec_root, backup_root)
                moved_paths.append(str(moved_path))
            action = "majority_moved" if moved_paths else "majority_only"
            logging.warning(
                "Majority face kept for %s (%s); moved %d non-majority photo(s)",
                ceo_name,
                company,
                len(moved_paths),
            )
        elif fallback_used:
            to_delete = [index_map[idx][1] for idx in index_map if idx not in canonical_idx and idx not in multi_idx]
            for path in to_delete:
                moved_path = move_with_structure(Path(path), sec_root, backup_root)
                moved_paths.append(str(moved_path))
            action = "fallback_latest_kept" if moved_paths else "fallback_no_other_photos"
            logging.warning(
                "Fallback triggered for %s (%s); kept idx %s and moved %d other photo(s)",
                ceo_name,
                company,
                canonical_idx,
                len(moved_paths),
            )
        elif verdict == "mixed" and confidence >= args.diff_threshold:
            to_delete = [index_map[idx][1] for idx in outlier_idx if idx in index_map]
            for path in to_delete:
                moved_path = move_with_structure(Path(path), sec_root, backup_root)
                moved_paths.append(str(moved_path))
            action = "outliers_moved" if moved_paths else "no_outliers_found"
            logging.warning(
                "Flagged %d outlier photo(s) for %s (%s)",
                len(moved_paths),
                ceo_name,
                company,
            )
        elif verdict == "mixed":
            action = "mixed_low_confidence"
        elif verdict == "same":
            action = "all_same_kept"

        majority_years = [str(index_map[idx][0]) for idx in majority_idx if idx in index_map]
        outlier_years = [str(index_map[idx][0]) for idx in outlier_idx if idx in index_map]
        multi_years = [str(index_map[idx][0]) for idx in multi_idx if idx in index_map]
        timestamp = datetime.utcnow().isoformat()
        report_rows.append(
            {
                "company": company,
                "cik": cik10,
                "ceo_name": ceo_name,
                "years": ",".join(str(year) for year, _ in available),
                "verdict": verdict,
                "confidence": f"{confidence:.4f}",
                "majority_indices": ",".join(str(idx) for idx in majority_idx),
                "majority_years": ",".join(majority_years),
                "outlier_indices": ",".join(str(idx) for idx in outlier_idx),
                "outlier_years": ",".join(outlier_years),
                "multi_indices": ",".join(str(idx) for idx in multi_idx),
                "multi_years": ",".join(multi_years),
                "fallback_used": "yes" if fallback_used else "no",
                "to_delete": "|".join(str(path) for path in to_delete),
                "action": action,
                "timestamp": timestamp,
            }
        )

    write_report(args.report, report_rows)
    logging.info("Wrote %d rows to %s", len(report_rows), args.report)


if __name__ == "__main__":
    main()


# python scrapephoto/verify_ceo_photos.py --csv scrapephoto/execucomp_company_identifiers_annual_2000_2025_with_ceo_FINAL.csv
