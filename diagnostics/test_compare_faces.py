import argparse
import logging
import sys
from pathlib import Path
from typing import List

from dashscope_vision import DashscopeVisionClient

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test helper: compare faces inside a directory via DashScope"
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("test"),
        help="Folder containing the photos to compare (default: ./test)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO)",
    )
    return parser.parse_args()


def collect_images(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder}")
    files: List[Path] = []
    for entry in sorted(folder.iterdir()):
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(entry.resolve())
    return files


def build_reason(
    verdict: str,
    majority: List[int],
    outliers: List[int],
    multi: List[int],
) -> str:
    majority_text = ",".join(str(idx) for idx in majority) if majority else "(none)"
    outlier_text = ",".join(str(idx) for idx in outliers) if outliers else "(none)"
    multi_text = ",".join(str(idx) for idx in multi) if multi else "(none)"

    if verdict == "multi" or multi:
        return (
            "Detected group photo(s); model still treats index "
            + majority_text
            + " as the most frequent face (CEO). Group photo indices: "
            + multi_text
        )

    if majority and len(majority) >= 2:
        prefix = f"Model sees indices {majority_text} as the dominant face(s), treated as the CEO."
        if outliers:
            prefix += f" Indices {outlier_text} appear to be different faces (not the CEO)."
        else:
            prefix += " No additional distinct faces detected."
        return prefix

    if not majority and outliers:
        return (
            "No repeated face detected; model flags these indices as not the same CEO: "
            + outlier_text
        )

    return "Model cannot determine the dominant face; manual review recommended."


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(message)s")
    try:
        images = collect_images(args.directory)
    except FileNotFoundError as exc:
        logging.error(str(exc))
        sys.exit(1)

    if len(images) < 2:
        logging.error("Need at least 2 photos in %s", args.directory)
        sys.exit(1)

    client = DashscopeVisionClient()
    if not client.is_ready():
        logging.error("DashScope API key not configured. Please set dashscope_api_key.txt")
        sys.exit(1)

    image_paths = [str(path) for path in images]
    labels = [path.name for path in images]
    verdict, confidence, majority, outliers, multi = client.compare_faces_detailed(
        image_paths, labels=labels
    )
    reason = build_reason(verdict, majority, outliers, multi)

    print("=== DashScope Test Result ===")
    print(f"Folder       : {args.directory}")
    print(f"Total photos : {len(images)}")
    print(f"Verdict      : {verdict}")
    print(f"Confidence   : {confidence:.4f}")
    print(f"Majority idx : {majority}")
    print(f"Outliers idx : {outliers}")
    print(f"Multi idx    : {multi}")
    print(f"Reason       : {reason}")
    print("\nIndex -> Filename mapping:")
    for idx, path in enumerate(images, start=1):
        print(f"  {idx}: {path.name}")


if __name__ == "__main__":
    main()
