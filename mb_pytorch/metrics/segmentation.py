"""Metrics for image segmentation tasks."""

from typing import Dict, List, Optional, Union
import torch
import numpy as np
from .base import BaseMetric

class SegmentationMetrics(BaseMetric):
    """Compute segmentation metrics including IoU and Dice coefficient."""
    
    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = None,
        threshold: float = 0.5
    ):
        """
        Initialize segmentation metrics.
        
        Args:
            num_classes: Number of classes
            ignore_index: Optional index to ignore in metrics
            threshold: Threshold for binary segmentation
        """
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.reset()
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update metrics with new predictions and targets.
        
        Args:
            preds: Predicted segmentation masks (N, C, H, W) or (N, H, W)
            targets: Ground truth masks (N, H, W)
        """
        if preds.dim() == 4:  # Multi-class case
            preds = preds.argmax(dim=1)
        elif preds.dim() == 3:  # Binary case
            preds = (preds > self.threshold).long()
            
        preds = preds.cpu()
        targets = targets.cpu()
        
        for pred, target in zip(preds, targets):
            self.predictions.append(pred)
            self.targets.append(target)
            
    def compute(self) -> Dict[str, float]:
        """
        Compute segmentation metrics.
        
        Returns:
            Dictionary with mean IoU, mean Dice, and per-class metrics
        """
        if not self.predictions:
            return {'mIoU': 0.0, 'mDice': 0.0}
            
        ious = []
        dices = []
        
        for c in range(self.num_classes):
            if c == self.ignore_index:
                continue
                
            iou = self._compute_iou_for_class(c)
            dice = self._compute_dice_for_class(c)
            
            if iou is not None:
                ious.append(iou)
                dices.append(dice)
                
        results = {
            'mIoU': np.mean(ious) if ious else 0.0,
            'mDice': np.mean(dices) if dices else 0.0
        }
        
        # Add per-class metrics
        for c, (iou, dice) in enumerate(zip(ious, dices)):
            if c != self.ignore_index:
                results[f'IoU_class_{c}'] = iou
                results[f'Dice_class_{c}'] = dice
                
        return results
    
    def _compute_iou_for_class(self, class_id: int) -> Optional[float]:
        """Compute IoU for a specific class."""
        total_intersection = 0
        total_union = 0
        
        for pred, target in zip(self.predictions, self.targets):
            pred_mask = (pred == class_id)
            target_mask = (target == class_id)
            
            intersection = (pred_mask & target_mask).sum().item()
            union = (pred_mask | target_mask).sum().item()
            
            total_intersection += intersection
            total_union += union
            
        return total_intersection / total_union if total_union > 0 else None
    
    def _compute_dice_for_class(self, class_id: int) -> Optional[float]:
        """Compute Dice coefficient for a specific class."""
        total_intersection = 0
        total_sum = 0
        
        for pred, target in zip(self.predictions, self.targets):
            pred_mask = (pred == class_id)
            target_mask = (target == class_id)
            
            intersection = (pred_mask & target_mask).sum().item()
            sum_total = pred_mask.sum().item() + target_mask.sum().item()
            
            total_intersection += 2 * intersection
            total_sum += sum_total
            
        return total_intersection / total_sum if total_sum > 0 else None
    
    def reset(self) -> None:
        """Reset metric state."""
        self.predictions = []
        self.targets = []


class BoundaryIoU(BaseMetric):
    """Compute boundary IoU for evaluating segmentation boundaries."""
    
    def __init__(
        self,
        num_classes: int,
        boundary_width: int = 1,
        ignore_index: Optional[int] = None
    ):
        """
        Initialize boundary IoU metric.
        
        Args:
            num_classes: Number of classes
            boundary_width: Width of boundary in pixels
            ignore_index: Optional index to ignore
        """
        super().__init__()
        self.num_classes = num_classes
        self.boundary_width = boundary_width
        self.ignore_index = ignore_index
        self.reset()
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update metric with new predictions and targets.
        
        Args:
            preds: Predicted segmentation masks (N, C, H, W) or (N, H, W)
            targets: Ground truth masks (N, H, W)
        """
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)
            
        preds = preds.cpu()
        targets = targets.cpu()
        
        for pred, target in zip(preds, targets):
            pred_boundaries = self._get_boundaries(pred)
            target_boundaries = self._get_boundaries(target)
            
            self.predictions.append(pred_boundaries)
            self.targets.append(target_boundaries)
            
    def compute(self) -> Dict[str, float]:
        """
        Compute boundary IoU metrics.
        
        Returns:
            Dictionary with mean boundary IoU and per-class values
        """
        if not self.predictions:
            return {'mBoundaryIoU': 0.0}
            
        ious = []
        for c in range(self.num_classes):
            if c == self.ignore_index:
                continue
                
            iou = self._compute_boundary_iou_for_class(c)
            if iou is not None:
                ious.append(iou)
                
        results = {'mBoundaryIoU': np.mean(ious) if ious else 0.0}
        
        # Add per-class metrics
        for c, iou in enumerate(ious):
            if c != self.ignore_index:
                results[f'BoundaryIoU_class_{c}'] = iou
                
        return results
    
    def _get_boundaries(self, mask: torch.Tensor) -> torch.Tensor:
        """Extract boundaries from segmentation mask."""
        from scipy.ndimage import binary_dilation
        
        boundaries = torch.zeros_like(mask)
        
        for c in range(self.num_classes):
            if c == self.ignore_index:
                continue
                
            class_mask = (mask == c).numpy()
            dilated = binary_dilation(class_mask, iterations=self.boundary_width)
            eroded = binary_dilation(class_mask, iterations=-self.boundary_width)
            boundary = torch.from_numpy(dilated ^ eroded)
            boundaries[boundary] = c
            
        return boundaries
    
    def _compute_boundary_iou_for_class(self, class_id: int) -> Optional[float]:
        """Compute boundary IoU for a specific class."""
        total_intersection = 0
        total_union = 0
        
        for pred, target in zip(self.predictions, self.targets):
            pred_boundary = (pred == class_id)
            target_boundary = (target == class_id)
            
            intersection = (pred_boundary & target_boundary).sum().item()
            union = (pred_boundary | target_boundary).sum().item()
            
            total_intersection += intersection
            total_union += union
            
        return total_intersection / total_union if total_union > 0 else None
    
    def reset(self) -> None:
        """Reset metric state."""
        self.predictions = []
        self.targets = []
