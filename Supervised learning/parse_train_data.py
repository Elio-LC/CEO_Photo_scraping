"""
Parse labeled data and convert it into a training-friendly format.
"""

from pathlib import Path
import json
from typing import List, Dict, Any


def parse_labeled_data(labeled_file: Path) -> List[Dict[str, Any]]:
    """Parse labeled data from disk."""
    if not labeled_file.exists():
        return []
    
    try:
        with open(labeled_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def convert_to_training_format(
    labeled_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Transform labeled samples into the training schema."""
    training_data = []
    
    for item in labeled_data:
        training_item = {
            'cik': item.get('cik'),
            'year': item.get('year'),
            'ceo': item.get('ceo'),
            'features': {
                'position_index': item.get('position_index', 0),
                'estimated_size': item.get('estimated_size', 0),
                'has_face': item.get('has_face', False),
            },
            'label': 1 if item.get('is_correct_ceo') else 0,
        }
        training_data.append(training_item)
    
    return training_data


def save_training_data(training_data: List[Dict[str, Any]], output_path: Path) -> None:
    """Persist training data to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
