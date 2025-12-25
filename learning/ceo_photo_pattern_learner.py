"""
Learn extraction patterns for CEO photos from human-labeled samples.
"""

from pathlib import Path
import json
from typing import Dict, List, Any


class CEOPhotoPatternLearner:
    """Learns patterns to improve CEO photo extraction."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def learn_from_samples(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Derive simple patterns from labeled samples."""
        patterns = {
            'common_positions': [],
            'common_stages': {},
            'success_rate_by_stage': {},
        }
        

        positions = [s.get('position_index', 0) for s in samples if s.get('success')]
        if positions:
            patterns['common_positions'] = sorted(positions)
        

        for sample in samples:
            stage = sample.get('stage', 'unknown')
            if stage not in patterns['common_stages']:
                patterns['common_stages'][stage] = 0
            if sample.get('success'):
                patterns['common_stages'][stage] += 1
        
        return patterns
    
    def save_patterns(self, patterns: Dict[str, Any], filename: str = "learned_patterns.json") -> None:
        """Persist learned patterns to disk."""
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(patterns, f, indent=2, ensure_ascii=False)
