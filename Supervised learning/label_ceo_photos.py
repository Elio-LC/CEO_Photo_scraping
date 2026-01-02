"""
Utilities for labeling and validating extracted CEO photos.
"""

from pathlib import Path
import json
from typing import List, Dict, Any


def load_samples(samples_file: Path) -> List[Dict[str, Any]]:
    """Load sample metadata if present."""
    if not samples_file.exists():
        return []
    
    try:
        with open(samples_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def save_labels(labels: List[Dict[str, Any]], output_path: Path) -> None:
    """Persist labeling results to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
