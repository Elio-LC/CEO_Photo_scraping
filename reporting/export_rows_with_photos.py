import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CEO_NAME_COLUMNS = [
    "ceo_name",
    "ceo_full_name",
    "ceo_fullname",
    "ceo",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a filtered CSV containing rows that already have at least one CEO photo in the pool."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("scrapephoto") / "execucomp_company_identifiers_annual_2000_2025_with_ceo_FINAL.csv",
        help="Source execucomp CSV dataset.",
    )
    parser.add_argument(
        "--sec-data",
        type=Path,
        default=Path("D:/ceo_photo_pool"),
        help="Root folder containing company ceophoto directories (default: D:/ceo_photo_pool).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("little.csv"),
        help="Destination CSV for rows with photos (default: ./little.csv).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(level=level.upper(), format="%(asctime)s %(levelname)s %(message)s")


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
    mapping: Dict[int, List[Path]] = {}
    if not directory.exists():
        return mapping
    for entry in directory.iterdir():
        if not entry.is_file() or entry.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = entry.stem
        digits: List[str] = []
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
        mapping.setdefault(key, []).append(entry)
    return mapping


def _index_pool_files_for_cik(sec_root: Path, cik10: str) -> Dict[int, List[Path]]:
    mapping: Dict[int, List[Path]] = {}
    pattern = f"{cik10}_*"
    for path in sec_root.glob(pattern):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        remainder = path.stem[len(cik10) + 1 :]
        digits: List[str] = []
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
        mapping.setdefault(year, []).append(path)
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
    keys = {year, year % 100}
    candidates: List[Path] = []
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


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    df = pd.read_csv(args.csv)
    if df.empty:
        logging.warning("Source CSV is empty: %s", args.csv)
        return

    sec_root = args.sec_data.resolve()
    dir_cache: Dict[Path, Dict[int, List[Path]]] = {}
    pool_cache: Dict[str, Dict[int, List[Path]]] = {}
    keep_indices: List[int] = []

    for idx, row in df.iterrows():
        cik_val = row.get("cik")
        company = str(row.get("compustat_name") or row.get("primary_company_name") or "").strip()
        fyear = row.get("fyear")
        ceo_name = extract_ceo_name(row)
        try:
            cik10 = f"{int(cik_val):010d}"
            year = int(fyear)
        except (TypeError, ValueError):
            continue
        if not company:
            continue
        ceo_dir = build_ceophoto_dir(sec_root, company, cik10)
        photo_path = find_photo_for_year(
            year,
            cik10,
            ceo_dir,
            sec_root,
            dir_cache,
            pool_cache,
        )
        if photo_path:
            keep_indices.append(idx)
            logging.debug(
                "Matched photo for %s (%s) year %s -> %s",
                ceo_name or "<unknown>",
                company,
                year,
                photo_path,
            )

    if not keep_indices:
        logging.warning("No rows with photos were found using pool %s", sec_root)
        return

    filtered = df.loc[keep_indices]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.output, index=False)
    logging.info("Wrote %d rows with photos to %s", len(filtered), args.output)


if __name__ == "__main__":
    main()
