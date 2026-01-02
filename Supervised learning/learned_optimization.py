"""
Learning-based optimizer that tunes parameters from extraction results.
"""

from pathlib import Path
import json
from typing import Dict, List, Any


class LearnedOptimization:
    """Optimizer that learns from historical run results."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.optimizations = {}
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize extraction performance for downstream tuning."""
        analysis = {
            'total': len(results),
            'success': 0,
            'success_by_stage': {},
            'avg_position_index': 0,
            'avg_estimated_size': 0,
        }
        
        success_results = [r for r in results if r.get('status') == 'success']
        analysis['success'] = len(success_results)
        
        if success_results:

            for result in success_results:
                stage = result.get('stage', 'unknown')
                if stage not in analysis['success_by_stage']:
                    analysis['success_by_stage'][stage] = 0
                analysis['success_by_stage'][stage] += 1
            

            positions = [r.get('position_index', 0) for r in success_results]
            sizes = [r.get('estimated_size', 0) for r in success_results]
            
            analysis['avg_position_index'] = sum(positions) / len(positions) if positions else 0
            analysis['avg_estimated_size'] = sum(sizes) / len(sizes) if sizes else 0
        
        return analysis
    
    def save_optimization(self, optimization: Dict[str, Any], filename: str = "learned_optimization.json") -> None:
        """Persist derived optimization settings."""
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(optimization, f, indent=2, ensure_ascii=False)
