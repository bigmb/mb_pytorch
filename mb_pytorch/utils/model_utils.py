"""Utilities for model manipulation and analysis."""

from typing import Dict, Any, Optional, Union, List
import torch
import torch.nn as nn
import onnx
from onnx2pytorch import ConvertModel
from torchsummary import summary

__all__ = [
    'ModelUtils',
    'FeatureExtractor'
]

class ModelUtils:
    """Utilities for model manipulation and analysis."""
    
    @staticmethod
    def get_model_summary(model: nn.Module, input_size: tuple) -> None:
        """
        Print model summary.
        
        Args:
            model: PyTorch model
            input_size: Input size tuple (channels, height, width)
        """
        summary(model, input_size=input_size)
    
    @staticmethod
    def onnx_to_pytorch(model_path: str) -> nn.Module:
        """
        Convert ONNX model to PyTorch.
        
        Args:
            model_path: Path to ONNX model file
            
        Returns:
            Converted PyTorch model
        """
        onnx_model = onnx.load(model_path)
        return ConvertModel(onnx_model)
    
    @staticmethod
    def overwrite_layer_weights(
        model: nn.Module,
        layer_index: int,
        new_weights: torch.Tensor,
        logger: Optional[Any] = None
    ) -> None:
        """
        Overwrite weights of a specific layer.
        
        Args:
            model: PyTorch model
            layer_index: Index of layer to modify
            new_weights: New weights tensor
            logger: Optional logger instance
        
        Raises:
            ValueError: If layer is not Conv2d or Linear
        """
        layer_name = list(model.named_modules())[layer_index][0]
        layer = getattr(model, layer_name)
        
        if logger:
            logger.info(f"Overwriting weights of layer {layer_name}")
            
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            layer.weight.data = new_weights
        else:
            raise ValueError("Layer must be Conv2d or Linear")


class FeatureExtractor:
    """Extract and visualize features from model layers."""
    
    def __init__(self, model: nn.Module):
        """
        Initialize feature extractor.
        
        Args:
            model: PyTorch model to extract features from
        """
        self.model = model
        
    def get_feature_extractor(self, layer_name: str) -> nn.Sequential:
        """
        Get feature extractor up to specified layer.
        
        Args:
            layer_name: Name of layer to extract features up to
            
        Returns:
            Sequential model up to specified layer
        """
        module = dict(self.model.named_modules())[layer_name]
        layer_idx = list(self.model.modules()).index(module)
        return nn.Sequential(*list(self.model.children())[:layer_idx])
    
    def extract_features(
        self,
        data: torch.Tensor,
        layer_names: List[str]
    ) -> List[torch.Tensor]:
        """
        Extract features from specified layers.
        
        Args:
            data: Input tensor
            layer_names: List of layer names to extract features from
            
        Returns:
            List of feature tensors
        """
        features = []
        for layer_name in layer_names:
            extractor = self.get_feature_extractor(layer_name)
            output = extractor(data)
            features.append(output.detach())
        return features
