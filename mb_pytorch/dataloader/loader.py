"""
PyTorch DataLoader implementation with support for classification, detection, and segmentation tasks.
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from mb_pytorch.utils.yaml_reader import YamlReader
from mb_pandas.src.dfload import load_any_df
from mb_utils.src.verify_image import verify_image
from mb.pandas import check_drop_duplicates, remove_unnamed
from ..utils.extra_utils import labels_num_map
from datetime import datetime

__all__ = ['BaseDataset', 'TorchDataLoader', 'TransformCompose']


class TransformCompose:
    """Compose multiple transforms with support for images, masks, and bounding boxes."""
    
    def __init__(self, transform_config: Dict[str, Any], logger: Optional[Any] = None):
        """
        Initialize transform composition.
        
        Args:
            transform_config: Transform configuration dictionary
            logger: Optional logger instance
        """
        self.config = transform_config
        self.logger = logger
        
        if not self.config.get('transform', True):
            if self.logger:
                self.logger.info("Transforms disabled in config")
    
    def __call__(
        self, 
        img: np.ndarray,
        mask: Optional[np.ndarray] = None,
        bbox: Optional[List[int]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply transforms to image and optionally mask or bbox."""
        
        # Convert to tensor first
        if self.config.get('to_tensor', {}).get('val', True):
            img = transforms.ToTensor()(img)
            if mask is not None:
                mask = transforms.ToTensor()(mask)
            if bbox is not None:
                bbox = torch.tensor(bbox, dtype=torch.int32)
        
        # Apply normalization
        if self.config.get('normalize', {}).get('val', False):
            norm_config = self.config['normalize']['args']
            img = transforms.Normalize(
                norm_config['mean'],
                norm_config['std']
            )(img)
        
        # Apply resize
        if self.config.get('resize', {}).get('val', False):
            orig_size = img.size()
            resize_size = self.config['resize']['args']['size']
            img = transforms.Resize(resize_size)(img)
            
            if mask is not None:
                mask = transforms.Resize(resize_size)(mask)
            if bbox is not None:
                bbox = self._resize_bbox(orig_size, img.size(), bbox)
        
        # Apply augmentations only if enabled
        if self.config.get('augment', True):
            img, mask, bbox = self._apply_augmentations(img, mask, bbox)
        
        if mask is not None:
            return img, mask
        elif bbox is not None:
            return img, bbox
        return img
    
    def _apply_augmentations(
        self,
        img: torch.Tensor,
        mask: Optional[torch.Tensor],
        bbox: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Apply configured augmentations to the data."""
        
        # Random horizontal flip
        if self.config.get('random_horizontal_flip', {}).get('val', False):
            if torch.rand(1) < self.config['random_horizontal_flip']['args']['p']:
                img = transforms.functional.hflip(img)
                if mask is not None:
                    mask = transforms.functional.hflip(mask)
                if bbox is not None:
                    bbox = self._hflip_bbox(bbox, img.size(2))
        
        # Random vertical flip
        if self.config.get('random_vertical_flip', {}).get('val', False):
            if torch.rand(1) < self.config['random_vertical_flip']['args']['p']:
                img = transforms.functional.vflip(img)
                if mask is not None:
                    mask = transforms.functional.vflip(mask)
                if bbox is not None:
                    bbox = self._vflip_bbox(bbox, img.size(1))
        
        # Color jitter
        if self.config.get('random_color_jitter', {}).get('val', False):
            jitter_config = self.config['random_color_jitter']['args']
            color_jitter = transforms.ColorJitter(
                brightness=jitter_config.get('brightness', 0),
                contrast=jitter_config.get('contrast', 0),
                saturation=jitter_config.get('saturation', 0),
                hue=jitter_config.get('hue', 0)
            )
            img = color_jitter(img)
        
        return img, mask, bbox
    
    def _resize_bbox(
        self,
        original_size: torch.Size,
        new_size: torch.Size,
        bbox: torch.Tensor
    ) -> torch.Tensor:
        """Resize bounding box coordinates."""
        _, orig_h, orig_w = original_size
        _, new_h, new_w = new_size
        
        x_scale = new_w / orig_w
        y_scale = new_h / orig_h
        
        bbox = bbox.clone()
        bbox[0::2] *= x_scale  # x coordinates
        bbox[1::2] *= y_scale  # y coordinates
        
        return bbox
    
    def _hflip_bbox(self, bbox: torch.Tensor, width: int) -> torch.Tensor:
        """Flip bounding box horizontally."""
        bbox = bbox.clone()
        bbox[0::2] = width - bbox[0::2]
        return bbox
    
    def _vflip_bbox(self, bbox: torch.Tensor, height: int) -> torch.Tensor:
        """Flip bounding box vertically."""
        bbox = bbox.clone()
        bbox[1::2] = height - bbox[1::2]
        return bbox


class BaseDataset(Dataset):
    """Base dataset class supporting classification, detection, and segmentation."""
    
    def __init__(
        self,
        data_config: Dict[str, Any],
        task_type: str,
        transform: Optional[TransformCompose] = None,
        is_train: bool = True,
        logger: Optional[Any] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_config: Dataset configuration
            task_type: Type of task ('classification', 'detection', 'segmentation')
            transform: Optional transform composition
            is_train: Whether this is training data
            logger: Optional logger instance
        """
        self.config = data_config
        self.task_type = task_type
        self.transform = transform
        self.is_train = is_train
        self.logger = logger
        
        # Load and process data
        self.data = self._load_data()
        self._verify_data()
        self._process_labels()
    
    def _load_data(self) -> pd.DataFrame:
        """Load and validate data from CSV."""
        if self.logger:
            self.logger.info(f"Loading data from {self.config['root']}")
            
        df = load_any_df(self.config['root'], logger=self.logger)
        
        required_cols = ['image_path', 'image_type', 'label']
        if self.task_type == 'segmentation':
            required_cols.append('mask_path')
        elif self.task_type == 'detection':
            required_cols.append('bbox')
            
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        return df
    
    def _verify_data(self):
        """Verify image paths and optionally image content."""
        # Verify paths exist
        self.data = self.data[self.data['image_path'].apply(os.path.exists)].reset_index(drop=True)
        
        # Optionally verify image content
        if self.config.get('verify_image', False):
            if self.logger:
                self.logger.info("Verifying images...")
            self.data = self.data[
                self.data['image_path'].apply(lambda x: verify_image(x, logger=self.logger))
            ].reset_index(drop=True)
        
        # Filter by split
        split_type = 'training' if self.is_train else 'validation'
        self.data = self.data[self.data['image_type'].isin([split_type, split_type[:4]])].reset_index(drop=True)
        
        # Remove duplicates and unnamed columns
        self.data = check_drop_duplicates(self.data, columns=['image_path'], drop=True, logger=self.logger)
        self.data = remove_unnamed(self.data, logger=self.logger)
    
    def _process_labels(self):
        """Process and map labels to numbers."""
        output_path = os.path.join(os.path.dirname(self.config['root']), 'label_num_map.csv')
        self.data = labels_num_map(self.data, output_csv=output_path)
        
        # Save processed data
        if os.path.dirname(self.config['root']):
            suffix = 'train' if self.is_train else 'val'
            self.data.to_csv(
                os.path.join(os.path.dirname(self.config['root']), f'{suffix}_wrangled_file.csv'),
                index=False
            )
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a data sample."""
        # Load image
        img_path = self.data['image_path'].iloc[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Prepare data based on task type
        if self.task_type == 'classification':
            if self.transform:
                img = self.transform(img)
            return {
                'image': img,
                'label': self.data['label_num'].iloc[idx]
            }
            
        elif self.task_type == 'segmentation':
            mask = cv2.imread(self.data['mask_path'].iloc[idx], cv2.IMREAD_GRAYSCALE)
            if self.transform:
                img, mask = self.transform(img, mask=mask)
            return {
                'image': img,
                'mask': mask,
                'label': self.data['label_num'].iloc[idx]
            }
            
        elif self.task_type == 'detection':
            bbox = eval(self.data['bbox'].iloc[idx])
            if self.transform:
                img, bbox = self.transform(img, bbox=bbox)
            else:
                img = transforms.ToTensor()(img)
                bbox = torch.tensor([bbox], dtype=torch.int32)
            return {
                'image': img,
                'bbox': bbox,
                'label': self.data['label_num'].iloc[idx]
            }


class TorchDataLoader(torch.utils.data.DataLoader):
    """DataLoader with support for classification, detection, and segmentation tasks."""
    
    def __init__(
        self,
        dataset: BaseDataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize DataLoader.
        
        Args:
            dataset: Dataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            collate_fn: Optional custom collate function
            **kwargs: Additional arguments to pass to torch DataLoader
        """
        if collate_fn is None:
            collate_fn = self._get_default_collate(dataset.task_type)
            
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            **kwargs
        )
    
    @staticmethod
    def _get_default_collate(task_type: str) -> Callable:
        """Get default collate function based on task type."""
        if task_type == 'classification':
            return lambda batch: {
                'image': torch.stack([b['image'] for b in batch]),
                'label': torch.tensor([b['label'] for b in batch])
            }
        elif task_type == 'segmentation':
            return lambda batch: {
                'image': torch.stack([b['image'] for b in batch]),
                'mask': torch.stack([b['mask'] for b in batch]),
                'label': torch.tensor([b['label'] for b in batch])
            }
        elif task_type == 'detection':
            return lambda batch: {
                'image': [b['image'] for b in batch],
                'bbox': torch.stack([b['bbox'] for b in batch]),
                'label': torch.tensor([b['label'] for b in batch])
            }
        else:
            raise ValueError(f"Unknown task type: {task_type}")
