"""
MB PyTorch - A PyTorch-based deep learning library for computer vision tasks
"""

from .classification.training import ClassificationTrainer, classification_train_loop
from .detection.training import DetectionTrainer, detection_train_loop
from .training.base_trainer import BaseTrainer
from .models.modelloader import ModelLoader, ModelExtractor

__all__ = [
    'ClassificationTrainer',
    'DetectionTrainer',
    'BaseTrainer',
    'ModelLoader',
    'ModelExtractor',
    'classification_train_loop',
    'detection_train_loop'
]

# __version__ = '0.1.0'  # Update this based on your versioning
