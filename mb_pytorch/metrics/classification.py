"""Metrics for classification tasks."""

from typing import Dict, Optional, List, Union
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from .base import BaseMetric, AverageMeter

class ClassificationMetrics(BaseMetric):
    """Compute various classification metrics."""
    
    def __init__(
        self,
        num_classes: int,
        average: str = 'macro',
        metrics: Optional[List[str]] = None
    ):
        """
        Initialize classification metrics.
        
        Args:
            num_classes: Number of classes
            average: Averaging method ('micro', 'macro', 'weighted')
            metrics: List of metrics to compute (default: all)
        """
        super().__init__()
        self.num_classes = num_classes
        self.average = average
        self.metrics = metrics or ['accuracy', 'precision', 'recall', 'f1']
        
        self.reset()
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update metrics with new predictions and targets.
        
        Args:
            preds: Predicted class probabilities or logits (N, C)
            targets: Ground truth labels (N,)
        """
        if preds.dim() > 1:
            preds = preds.argmax(dim=1)
            
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        
        self.predictions.extend(preds)
        self.targets.extend(targets)
        
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        if not self.predictions:
            return {metric: 0.0 for metric in self.metrics}
            
        results = {}
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        if 'accuracy' in self.metrics:
            results['accuracy'] = accuracy_score(targets, preds)
            
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets,
            preds,
            average=self.average,
            zero_division=0
        )
        
        if 'precision' in self.metrics:
            results['precision'] = precision
        if 'recall' in self.metrics:
            results['recall'] = recall
        if 'f1' in self.metrics:
            results['f1'] = f1
            
        return results
    
    def reset(self) -> None:
        """Reset metric state."""
        self.predictions: List[int] = []
        self.targets: List[int] = []


class TopKAccuracy(BaseMetric):
    """Compute top-k accuracy."""
    
    def __init__(self, k: Union[int, List[int]] = 1):
        """
        Initialize top-k accuracy metric.
        
        Args:
            k: k value or list of k values
        """
        super().__init__()
        self.k = [k] if isinstance(k, int) else k
        self.max_k = max(self.k)
        self.reset()
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update metric with new predictions and targets.
        
        Args:
            preds: Predicted class probabilities or logits (N, C)
            targets: Ground truth labels (N,)
        """
        batch_size = targets.size(0)
        
        _, pred = preds.topk(self.max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        for k in self.k:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            self.correct[k] += correct_k.item()
            self.total[k] += batch_size
            
    def compute(self) -> Union[float, Dict[str, float]]:
        """
        Compute top-k accuracy values.
        
        Returns:
            Single accuracy value if k is int, dict of values if k is list
        """
        if len(self.k) == 1:
            k = self.k[0]
            return (self.correct[k] / self.total[k]) * 100.0
            
        return {
            f'top{k}_acc': (self.correct[k] / self.total[k]) * 100.0
            for k in self.k
        }
    
    def reset(self) -> None:
        """Reset metric state."""
        self.correct = {k: 0 for k in self.k}
        self.total = {k: 0 for k in self.k}


class ConfusionMatrix(BaseMetric):
    """Compute confusion matrix."""
    
    def __init__(self, num_classes: int, normalize: bool = False):
        """
        Initialize confusion matrix metric.
        
        Args:
            num_classes: Number of classes
            normalize: Whether to normalize matrix
        """
        super().__init__()
        self.num_classes = num_classes
        self.normalize = normalize
        self.reset()
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update confusion matrix with new predictions and targets.
        
        Args:
            preds: Predicted class probabilities or logits (N, C)
            targets: Ground truth labels (N,)
        """
        if preds.dim() > 1:
            preds = preds.argmax(dim=1)
            
        for p, t in zip(preds, targets):
            self.matrix[t.long(), p.long()] += 1
            
    def compute(self) -> torch.Tensor:
        """
        Compute confusion matrix.
        
        Returns:
            Confusion matrix tensor
        """
        if self.normalize:
            return self.matrix / self.matrix.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return self.matrix
    
    def reset(self) -> None:
        """Reset metric state."""
        self.matrix = torch.zeros(self.num_classes, self.num_classes)
