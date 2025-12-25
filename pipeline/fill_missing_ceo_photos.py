import argparse
import logging
import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CEO_NAME_COLUMNS = ["ceo_name", "ceo_full_name", "ceo_fullname", "ceo"]
NAME_NOISE_TOKENS = {"mr", "mrs", "ms", "dr", "jr", "sr", "ii", "iii", "iv", "v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill missing CEO photo years by copying the closest existing year for the same company/CEO."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("scrapephoto") / "execucomp_company_identifiers_annual_2000_2025_with_ceo_FINAL.csv",
        help="Execucomp CSV path (default: scrapephoto/...FINAL.csv)",
    )
    parser.add_argument(
        "--sec-data",
        type=Path,
        default=Path("D:/ceo_photo_pool"),
        help="Root directory containing <company>_<cik>/ceophoto folders (default: D:/ceo_photo_pool)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only log planned operations without copying any files.",
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
        digits: List[str] = []
        for ch in entry.stem:
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


def choose_reference_year(target_year: int, available_years: List[int]) -> Optional[int]:
    if not available_years:
        return None
    return min(available_years, key=lambda year: (abs(year - target_year), -year))


def build_destination_name(
    reference_path: Path,
    target_year: int,
    cik10: str,
    ceo_dir: Path,
) -> Path:
    if reference_path.is_relative_to(ceo_dir):
        digits: List[str] = []
        for ch in reference_path.stem:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if digits:
            prefix = str(target_year) if len(digits) >= 3 else f"{target_year % 100:02d}"
            suffix_part = reference_path.stem[len(digits) :]
            stem = prefix + suffix_part + "_fill"
        else:
            stem = f"{target_year}_{reference_path.stem}_fill"
        candidate = ceo_dir / f"{stem}{reference_path.suffix}"
    elif reference_path.name.startswith(f"{cik10}_"):
        candidate = reference_path.parent / f"{cik10}_{target_year}_fill{reference_path.suffix}"
    else:
        candidate = reference_path.parent / f"{target_year}_{reference_path.stem}_fill{reference_path.suffix}"

    counter = 1
    base_candidate = candidate
    while candidate.exists():
        candidate = base_candidate.with_name(f"{base_candidate.stem}_{counter}{base_candidate.suffix}")
        counter += 1
    return candidate


def copy_photo(source: Path, destination: Path, dry_run: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        logging.info("[DRY-RUN] would copy %s -> %s", source, destination)
        return
    shutil.copy2(source, destination)
    logging.info("Copied %s -> %s", source, destination)


def is_name_similar(name1: str, name2: str) -> bool:
    """
    Relaxed-but-safe similarity for CEO names so we can merge small variants.
    Rules (stop early on first match):
    1) Token subset (handles suffix like "John Smith CPA").
    2) Same last name AND
       a) same first token, OR
       b) first tokens share a >=3-char prefix, OR
       c) first tokens very similar (SequenceMatcher), OR
       d) one side is only an initial and middles align.
    """
    def _normalize(name: str) -> List[str]:
        # Split on non-alphanumerics, drop honorifics/suffixes, keep order
        tokens = [t for t in re.split(r"[^\w]+", name.lower()) if t]
        return [t for t in tokens if t not in NAME_NOISE_TOKENS]

    def _primary_first_token(tokens: List[str]) -> str:
        for token in tokens:
            if len(token) > 1:
                return token
        return tokens[0] if tokens else ""

    tokens1 = _normalize(name1)
    tokens2 = _normalize(name2)
    if not tokens1 or not tokens2:
        return False

    set1 = set(tokens1)
    set2 = set(tokens2)
    if set1.issubset(set2) or set2.issubset(set1):
        return True

    last1 = tokens1[-1]
    last2 = tokens2[-1]
    if last1 != last2:
        return False

    first1 = _primary_first_token(tokens1)
    first2 = _primary_first_token(tokens2)
    if first1 == first2:
        return True

    # Prefix overlap like "tim" vs "timothy"
    if (first1.startswith(first2) or first2.startswith(first1)) and min(len(first1), len(first2)) >= 3:
        return True

    # Small edit/nickname distance safeguard
    if SequenceMatcher(None, first1, first2).ratio() >= 0.82:
        return True

    # Initial-only first name with matching middle tokens
    if first1[0] == first2[0] and (len(first1) <= 2 or len(first2) <= 2):
        middle1 = {t for t in tokens1[1:-1] if len(t) > 1}
        middle2 = {t for t in tokens2[1:-1] if len(t) > 1}
        if not middle1 or not middle2 or middle1.issubset(middle2) or middle2.issubset(middle1):
            return True

    return False


def group_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    # First pass: collect all records by CIK
    cik_groups: Dict[str, List[Dict[str, Any]]] = {}
    
    for _, row in df.iterrows():
        ceo_name = extract_ceo_name(row)
        if not ceo_name:
            continue
        cik_val = row.get("cik")
        company = str(row.get("compustat_name") or row.get("primary_company_name") or "").strip()
        if not company:
            continue
        fyear = row.get("fyear")
        try:
            cik10 = f"{int(cik_val):010d}"
            year_int = int(fyear)
        except (TypeError, ValueError):
            continue
            
        if cik10 not in cik_groups:
            cik_groups[cik10] = []
            
        cik_groups[cik10].append({
            "cik10": cik10,
            "ceo_name": ceo_name,
            "company": company,
            "year": year_int
        })

    final_groups: List[Dict[str, Any]] = []

    # Second pass: cluster names within each CIK
    for cik10, records in cik_groups.items():
        # Get unique names
        unique_names = sorted(list({r["ceo_name"] for r in records}), key=len)
        
        # Cluster names
        # Map each name to a "canonical" name (the representative of the cluster)
        name_map: Dict[str, str] = {}
        clusters: List[str] = [] # List of canonical names
        
        for name in unique_names:
            matched_canonical = None
            for canonical in clusters:
                if is_name_similar(name, canonical):
                    matched_canonical = canonical
                    break
            
            if matched_canonical:
                name_map[name] = matched_canonical
            else:
                clusters.append(name)
                name_map[name] = name
        
        # Build final groups based on canonical names
        grouped_by_canonical: Dict[str, Dict[str, Any]] = {}
        
        for r in records:
            original_name = r["ceo_name"]
            canonical_name = name_map[original_name]
            
            if canonical_name not in grouped_by_canonical:
                grouped_by_canonical[canonical_name] = {
                    "cik10": cik10,
                    "ceo_name": canonical_name, # Use canonical name for the group
                    "company": r["company"],
                    "years": set(),
                    "original_names": set() # Keep track of variations
                }
            
            grouped_by_canonical[canonical_name]["years"].add(r["year"])
            grouped_by_canonical[canonical_name]["original_names"].add(original_name)
            
        final_groups.extend(grouped_by_canonical.values())
        
    return final_groups


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    if df.empty:
        logging.warning("CSV %s is empty", args.csv)
        return

    sec_root = args.sec_data.resolve()
    dir_cache: Dict[Path, Dict[int, List[Path]]] = {}
    pool_cache: Dict[str, Dict[int, List[Path]]] = {}
    total_filled = 0

    groups = group_records(df)
    for group in groups:
        cik10 = group["cik10"]
        company = group["company"]
        ceo_name = group["ceo_name"]
        ceo_dir = build_ceophoto_dir(sec_root, company, cik10)
        years = sorted(group["years"])

        available_by_year: Dict[int, Path] = {}
        for year in years:
            path = find_photo_for_year(year, cik10, ceo_dir, sec_root, dir_cache, pool_cache)
            if path is not None:
                available_by_year[year] = path

        missing_years = [year for year in years if year not in available_by_year]
        if not missing_years:
            continue

        existing_years = sorted(available_by_year.keys())
        if not existing_years:
            continue

        logging.info(
            "Processing %s (%s - %s): %s missing year(s)",
            company,
            cik10,
            ceo_name,
            len(missing_years),
        )

        for year in missing_years:
            ref_year = choose_reference_year(year, existing_years)
            if ref_year is None:
                continue
            source_path = available_by_year[ref_year]
            destination_path = build_destination_name(source_path, year, cik10, ceo_dir)
            if destination_path.exists():
                logging.info(
                    "Skipping %s (%s) year %s -> destination exists: %s",
                    company,
                    cik10,
                    year,
                    destination_path,
                )
                continue
            copy_photo(source_path, destination_path, args.dry_run)
            total_filled += 1
            available_by_year[year] = destination_path
            existing_years = sorted(available_by_year.keys())

    logging.info("Completed fill operation. Total photos created: %s", total_filled)


if __name__ == "__main__":
    main()
#python scrapephoto/fill_missing_ceo_photos.py
