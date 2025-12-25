#!/usr/bin/env python3
"""
Deduplicate photos in ceo_photo_pool, keeping the oldest (earliest mtime) copy.

Duplicates are defined as multiple files for the same CIK + year, such as:
  - 0000001234_2021.png
  - 0000001234_2021_fill.png
  - 0000001234_2021 (1).png

Usage:
  python dedupe_photos.py --pool-dir D:/ceo_photo_pool
  python dedupe_photos.py --pool-dir D:/ceo_photo_pool --dry-run
"""

import argparse
import os
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def parse_photo_key(filename: str) -> str | None:
    """
    Extract the CIK + year key from a filename.

    Supported formats:
      - 0000001234_2021.png       -> "0000001234_2021"
      - 0000001234_2021_fill.png  -> "0000001234_2021"
      - 0000001234_2021 (1).png   -> "0000001234_2021"
    """

    match = re.match(r'^(\d{10}_\d{4})', filename)
    if match:
        return match.group(1)
    return None


def find_duplicates(pool_dir: Path) -> dict[str, list[Path]]:
    """
    Scan the directory and return duplicated photo groups.

    Returns {key: [file1, file2, ...]} where len(files) > 1 indicates duplicates.
    """
    photo_groups: dict[str, list[Path]] = defaultdict(list)
    
    for file_path in pool_dir.glob("*.png"):
        if not file_path.is_file():
            continue
        
        key = parse_photo_key(file_path.name)
        if key:
            photo_groups[key].append(file_path)
    

    duplicates = {k: v for k, v in photo_groups.items() if len(v) > 1}
    return duplicates


def get_mtime(file_path: Path) -> float:
    """Return the file modification timestamp."""
    return file_path.stat().st_mtime


def dedupe_photos(pool_dir: Path, dry_run: bool = True) -> tuple[int, int]:
    """
    Deduplicate photos, keeping the earliest modified file.

    Args:
        pool_dir: Photo directory.
        dry_run: If True, print actions without deleting.

    Returns:
        (kept_count, deleted_count)
    """
    duplicates = find_duplicates(pool_dir)
    
    if not duplicates:
        print("No duplicate photos found.")
        return 0, 0
    
    print(f"Found {len(duplicates)} duplicate groups:\n")
    
    kept_count = 0
    deleted_count = 0
    
    for key, files in sorted(duplicates.items()):

        files_sorted = sorted(files, key=get_mtime)
        
        keep_file = files_sorted[0]
        delete_files = files_sorted[1:]
        
        keep_mtime = datetime.fromtimestamp(get_mtime(keep_file))
        
        print(f"[{key}]")
        print(f"  Keep: {keep_file.name} (mtime: {keep_mtime:%Y-%m-%d %H:%M:%S})")
        
        for del_file in delete_files:
            del_mtime = datetime.fromtimestamp(get_mtime(del_file))
            print(f"  Delete: {del_file.name} (mtime: {del_mtime:%Y-%m-%d %H:%M:%S})")
            
            if not dry_run:
                try:
                    del_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"    Warning: failed to delete ({e})")
            else:
                deleted_count += 1
        
        kept_count += 1
        print()
    
    return kept_count, deleted_count


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate CEO photos, keeping the earliest modified file"
    )
    parser.add_argument(
        "--pool-dir",
        type=Path,
        default=Path("D:/ceo_photo_pool"),
        help="Photo directory path (default: D:/ceo_photo_pool)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only; do not delete files",
    )
    
    args = parser.parse_args()
    
    if not args.pool_dir.exists():
        print(f"Error: directory not found - {args.pool_dir}")
        return 1
    
    if args.dry_run:
        print("=" * 60)
        print("[Preview mode] Files will not be deleted")
        print("=" * 60)
        print()
    else:
        print("=" * 60)
        print("[Execution mode] Duplicate files will be deleted")
        print("=" * 60)
        print()
    
    kept, deleted = dedupe_photos(args.pool_dir, dry_run=args.dry_run)
    
    print("=" * 60)
    if args.dry_run:
        print(f"Preview: {kept} duplicate groups; would keep {kept} and delete {deleted}")
        print("Remove --dry-run to perform deletion")
    else:
        print(f"Done: kept {kept}, deleted {deleted}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
