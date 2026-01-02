"""
Feedback Supervised learning module to improve extraction strategies from human review.
"""

from pathlib import Path
import json
from typing import Dict, List, Any, Optional


class FeedbackLearner:
    """Learns from human feedback to adjust extraction."""
    
    def __init__(self, feedback_file: Path):
        self.feedback_file = Path(feedback_file)
        self.feedback = []
        self.load()
    
    def load(self) -> None:
        """Load existing feedback entries."""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    self.feedback = json.load(f)
            except Exception:
                self.feedback = []
    
    def add_feedback(
        self,
        cik10: str,
        year: int,
        ceo_name: str,
        correct: bool,
        reason: Optional[str] = None,
    ) -> None:
        """Add a feedback record."""
        feedback_item = {
            'cik10': cik10,
            'year': year,
            'ceo_name': ceo_name,
            'correct': correct,
            'reason': reason,
        }
        self.feedback.append(feedback_item)
        self.save()
    
    def save(self) -> None:
        """Persist feedback to disk."""
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.feedback_file, 'w', encoding='utf-8') as f:
            json.dump(self.feedback, f, indent=2, ensure_ascii=False)
    
    def get_accuracy(self) -> float:
        """Compute accuracy based on stored feedback."""
        if not self.feedback:
            return 0.0
        
        correct = sum(1 for f in self.feedback if f.get('correct'))
        return correct / len(self.feedback)
