"""
Training and testing helpers for CEO photo extraction heuristics.
"""

from pathlib import Path
import json
from typing import List, Dict, Any, Tuple


class CEOPhotoModel:
    """Lightweight CEO photo extraction model."""
    
    def __init__(self):
        self.model_params = {}
    
    def train(self, training_data: List[Dict[str, Any]]) -> None:
        """Fit simple aggregate parameters from training data."""

        self.model_params = {
            'avg_position': self._calculate_avg('position_index', training_data),
            'avg_size': self._calculate_avg('estimated_size', training_data),
            'success_rate': self._calculate_success_rate(training_data),
        }
    
    def predict(self, features: Dict[str, Any]) -> float:
        """Predict confidence (0-1) that a candidate is the correct CEO photo."""
        if not self.model_params:
            return 0.5
        

        position_score = 1.0 if abs(
            features.get('position_index', 0) - self.model_params['avg_position']
        ) < 100 else 0.5
        
        size_score = 1.0 if abs(
            features.get('estimated_size', 0) - self.model_params['avg_size']
        ) < 1_000_000 else 0.5
        
        return (position_score + size_score) / 2.0
    
    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate accuracy on test data."""
        correct = 0
        for item in test_data:
            prediction = self.predict(item.get('features', {}))
            actual = item.get('label', 0)
            if (prediction > 0.5 and actual == 1) or (prediction <= 0.5 and actual == 0):
                correct += 1
        
        accuracy = correct / len(test_data) if test_data else 0.0
        return {'accuracy': accuracy, 'correct': correct, 'total': len(test_data)}
    
    @staticmethod
    def _calculate_avg(key: str, data: List[Dict[str, Any]]) -> float:
        """Compute an average value for a key."""
        values = [item.get(key, 0) for item in data if key in item]
        return sum(values) / len(values) if values else 0.0
    
    @staticmethod
    def _calculate_success_rate(data: List[Dict[str, Any]]) -> float:
        """Compute success rate based on label==1."""
        success = sum(1 for item in data if item.get('label') == 1)
        return success / len(data) if data else 0.0


def split_data(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.8,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split data into train and test portions."""
    split_point = int(len(data) * train_ratio)
    return data[:split_point], data[split_point:]


def train_and_evaluate(
    training_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Train the model and return evaluation metrics."""
    model = CEOPhotoModel()
    model.train(training_data)
    results = model.evaluate(test_data)
    return {
        'model_params': model.model_params,
        'evaluation_results': results,
    }
