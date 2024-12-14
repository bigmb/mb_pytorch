"""
MB PyTorch metrics for model evaluation.

This package provides metrics for evaluating model performance:
- Classification metrics (accuracy, precision, recall, F1)
- Detection metrics (mAP, IoU)
- Segmentation metrics (IoU, Dice coefficient)
- Loss functions for different tasks
"""

from .classification import ClassificationMetrics
from .detection import DetectionMetrics
from .segmentation import SegmentationMetrics
from .losses import LossFactory

__all__ = [
    'ClassificationMetrics',
    'DetectionMetrics',
    'SegmentationMetrics',
    'LossFactory'
]
