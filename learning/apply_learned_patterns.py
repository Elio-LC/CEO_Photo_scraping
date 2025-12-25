"""
Apply learned extraction patterns to improve current ranking and filtering.
"""

from pathlib import Path
import json
from typing import Dict, List, Any, Optional


def load_learned_patterns(patterns_file: Path) -> Dict[str, Any]:
    """Load previously learned patterns from disk."""
    if not patterns_file.exists():
        return {}
    
    try:
        with open(patterns_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def apply_patterns(
    candidates: List[Dict[str, Any]],
    patterns: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Re-rank or filter candidates using learned patterns."""
    if not patterns:
        return candidates
    

    common_positions = patterns.get('common_positions', [])
    if common_positions:
        def position_score(candidate: Dict[str, Any]) -> int:
            pos = candidate.get('position_index', 0)
            if pos in common_positions:
                return common_positions.index(pos)
            return len(common_positions)
        
        candidates = sorted(candidates, key=position_score)
    
    return candidates
