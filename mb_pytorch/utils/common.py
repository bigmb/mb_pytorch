"""Common utility functions used across the package."""

from typing import Dict, Any, Optional, Union, List
import pandas as pd
import numpy as np
import torch
from pathlib import Path

__all__ = [
    'LabelMapper',
    'PathUtils',
    'TensorUtils'
]

class LabelMapper:
    """Utilities for mapping labels to numbers and vice versa."""
    
    @staticmethod
    def create_label_mapping(
        data: Union[pd.DataFrame, str],
        label_column: str = 'label',
        output_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Create mapping between labels and numeric indices.
        
        Args:
            data: DataFrame or path to CSV
            label_column: Name of label column
            output_path: Optional path to save mapping
            
        Returns:
            DataFrame with original data and numeric labels
        """
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()
            
        # Convert labels to list format if they're strings
        labels_list = df[label_column].tolist()
        labels_list = [[i] for i in labels_list if isinstance(i, str)]
        
        # Create unique label mapping
        unique_labels = list(set([label for labels in labels_list for label in labels]))
        label_num_map = {label: i for i, label in enumerate(unique_labels)}
        
        # Create mapping DataFrame
        mapping_df = pd.DataFrame({
            'label': list(label_num_map.keys()),
            'label_num': list(label_num_map.values())
        })
        
        # Save mapping if path provided
        if output_path:
            mapping_df.to_csv(output_path, index=False)
        
        # Map labels to numbers in original DataFrame
        numeric_labels = []
        for label_list in labels_list:
            temp_list = [label_num_map[label] for label in label_list]
            numeric_labels.extend(temp_list)
            
        df['label_num'] = numeric_labels
        return df


class PathUtils:
    """Utilities for path handling."""
    
    @staticmethod
    def ensure_path_exists(path: Union[str, Path]) -> Path:
        """
        Ensure path exists, create if necessary.
        
        Args:
            path: Path to check/create
            
        Returns:
            Path object
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def get_project_root() -> Path:
        """
        Get project root directory.
        
        Returns:
            Path to project root
        """
        return Path(__file__).parent.parent.parent


class TensorUtils:
    """Utilities for tensor manipulation."""
    
    @staticmethod
    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor to numpy array.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Numpy array
        """
        return tensor.cpu().detach().numpy()
    
    @staticmethod
    def to_device(
        data: Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]],
        device: torch.device
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]:
        """
        Move data to device.
        
        Args:
            data: Input data (tensor, dict of tensors, or list of tensors)
            device: Target device
            
        Returns:
            Data on target device
        """
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in data.items()}
        elif isinstance(data, list):
            return [d.to(device) if isinstance(d, torch.Tensor) else d 
                   for d in data]
        return data
    
    @staticmethod
    def normalize_tensor(
        tensor: torch.Tensor,
        mean: Union[float, List[float]] = 0.0,
        std: Union[float, List[float]] = 1.0
    ) -> torch.Tensor:
        """
        Normalize tensor.
        
        Args:
            tensor: Input tensor
            mean: Mean for normalization
            std: Standard deviation for normalization
            
        Returns:
            Normalized tensor
        """
        if isinstance(mean, (int, float)):
            mean = [mean] * tensor.shape[1]
        if isinstance(std, (int, float)):
            std = [std] * tensor.shape[1]
            
        mean = torch.tensor(mean, device=tensor.device).view(1, -1, 1, 1)
        std = torch.tensor(std, device=tensor.device).view(1, -1, 1, 1)
        
        return (tensor - mean) / std
