"""Loss functions for various tasks."""

from typing import Dict, Optional, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

class LossFactory:
    """Factory class for creating loss functions."""
    
    @staticmethod
    def create_loss(loss_type: str, **kwargs) -> nn.Module:
        """
        Create loss function by name.
        
        Args:
            loss_type: Name of loss function
            **kwargs: Additional arguments for loss function
            
        Returns:
            Loss function module
            
        Raises:
            ValueError: If loss type is unknown
        """
        loss_map = {
            # Classification losses
            'cross_entropy': CrossEntropyLoss,
            'focal': FocalLoss,
            'label_smoothing': LabelSmoothingLoss,
            
            # Detection losses
            'giou': GIoULoss,
            'smooth_l1': SmoothL1Loss,
            
            # Segmentation losses
            'dice': DiceLoss,
            'tversky': TverskyLoss,
            'focal_tversky': FocalTverskyLoss
        }
        
        if loss_type not in loss_map:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
        return loss_map[loss_type](**kwargs)


class CrossEntropyLoss(nn.Module):
    """Cross entropy loss with optional class weights."""
    
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = 'mean'
    ):
        """
        Initialize cross entropy loss.
        
        Args:
            weight: Optional class weights
            ignore_index: Index to ignore
            reduction: Reduction method
        """
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction
        )
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        return self.loss_fn(inputs, targets)


class FocalLoss(nn.Module):
    """Focal loss for dealing with class imbalance."""
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better generalization."""
    
    def __init__(self, smoothing: float = 0.1):
        """
        Initialize label smoothing loss.
        
        Args:
            smoothing: Smoothing factor
        """
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        log_prob = F.log_softmax(inputs, dim=-1)
        num_classes = inputs.size(-1)
        
        # Create smoothed labels
        smooth_targets = torch.zeros_like(log_prob)
        smooth_targets.fill_(self.smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        return (-smooth_targets * log_prob).sum(dim=-1).mean()


class GIoULoss(nn.Module):
    """Generalized IoU loss for bounding box regression."""
    
    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss."""
        pred_x1 = pred_boxes[..., 0]
        pred_y1 = pred_boxes[..., 1]
        pred_x2 = pred_boxes[..., 2]
        pred_y2 = pred_boxes[..., 3]
        
        target_x1 = target_boxes[..., 0]
        target_y1 = target_boxes[..., 1]
        target_x2 = target_boxes[..., 2]
        target_y2 = target_boxes[..., 3]
        
        # Intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                    torch.clamp(inter_y2 - inter_y1, min=0)
                    
        # Union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area
        
        # Enclosing box
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
        
        # GIoU
        iou = inter_area / union_area
        giou = iou - (enclose_area - union_area) / enclose_area
        
        return 1 - giou.mean()


class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""
    
    def __init__(self, smooth: float = 1.0):
        """
        Initialize Dice loss.
        
        Args:
            smooth: Smoothing factor
        """
        super().__init__()
        self.smooth = smooth
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        inputs = torch.sigmoid(inputs)
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / \
               (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class TverskyLoss(nn.Module):
    """Tversky loss for handling class imbalance in segmentation."""
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7):
        """
        Initialize Tversky loss.
        
        Args:
            alpha: False positive weight
            beta: False negative weight
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        inputs = torch.sigmoid(inputs)
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        tp = (inputs * targets).sum()
        fp = (inputs * (1 - targets)).sum()
        fn = ((1 - inputs) * targets).sum()
        
        tversky = tp / (tp + self.alpha * fp + self.beta * fn)
        
        return 1 - tversky


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss combining focal and Tversky losses."""
    
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75
    ):
        """
        Initialize Focal Tversky loss.
        
        Args:
            alpha: False positive weight
            beta: False negative weight
            gamma: Focal parameter
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        inputs = torch.sigmoid(inputs)
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        tp = (inputs * targets).sum()
        fp = (inputs * (1 - targets)).sum()
        fn = ((1 - inputs) * targets).sum()
        
        tversky = tp / (tp + self.alpha * fp + self.beta * fn)
        focal_tversky = (1 - tversky) ** self.gamma
        
        return focal_tversky


class SmoothL1Loss(nn.Module):
    """Smooth L1 loss for bounding box regression."""
    
    def __init__(self, beta: float = 1.0):
        """
        Initialize Smooth L1 loss.
        
        Args:
            beta: Threshold for switching between L1 and L2 loss
        """
        super().__init__()
        self.beta = beta
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        diff = torch.abs(inputs - targets)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        return loss.mean()
