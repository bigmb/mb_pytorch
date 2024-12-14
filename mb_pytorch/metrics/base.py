"""Base classes for metrics computation."""

from typing import Dict, Any, Optional, Union, List
import torch
import numpy as np
from abc import ABC, abstractmethod

class BaseMetric(ABC):
    """Abstract base class for all metrics."""
    
    def __init__(self):
        self.reset()
    
    @abstractmethod
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update metric with new predictions and targets.
        
        Args:
            preds: Model predictions
            targets: Ground truth targets
        """
        pass
    
    @abstractmethod
    def compute(self) -> Union[float, Dict[str, float]]:
        """
        Compute final metric value(s).
        
        Returns:
            Metric value or dictionary of metric values
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset metric state."""
        pass


class AverageMeter:
    """Keep running average of a quantity."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset meter."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val: float, n: int = 1):
        """
        Update meter with new value.
        
        Args:
            val: Value to add
            n: Number of items this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricCollection:
    """Collection of metrics that can be updated together."""
    
    def __init__(self, metrics: Dict[str, BaseMetric]):
        """
        Initialize metric collection.
        
        Args:
            metrics: Dictionary of named metrics
        """
        self.metrics = metrics
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update all metrics with new predictions and targets.
        
        Args:
            preds: Model predictions
            targets: Ground truth targets
        """
        for metric in self.metrics.values():
            metric.update(preds, targets)
            
    def compute(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        return {name: metric.compute() 
                for name, metric in self.metrics.items()}
    
    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()


class MovingAverageMetric(BaseMetric):
    """Base class for metrics that maintain a moving average."""
    
    def __init__(self, window_size: Optional[int] = None):
        """
        Initialize moving average metric.
        
        Args:
            window_size: Optional size of moving average window
        """
        super().__init__()
        self.window_size = window_size
        self.values: List[float] = []
        
    def update(self, value: float) -> None:
        """
        Update metric with new value.
        
        Args:
            value: New value to add
        """
        self.values.append(value)
        if self.window_size and len(self.values) > self.window_size:
            self.values.pop(0)
            
    def compute(self) -> float:
        """
        Compute current average.
        
        Returns:
            Moving average value
        """
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)
    
    def reset(self) -> None:
        """Reset metric state."""
        self.values = []
