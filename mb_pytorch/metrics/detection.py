"""Metrics for object detection tasks."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from .base import BaseMetric

class DetectionMetrics(BaseMetric):
    """Compute object detection metrics including mAP and IoU."""
    
    def __init__(
        self,
        num_classes: int,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.5
    ):
        """
        Initialize detection metrics.
        
        Args:
            num_classes: Number of classes
            iou_threshold: IoU threshold for matching predictions to ground truth
            score_threshold: Confidence score threshold for predictions
        """
        super().__init__()
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.reset()
        
    def update(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]]
    ) -> None:
        """
        Update metrics with new predictions and targets.
        
        Args:
            predictions: List of prediction dicts with 'boxes', 'labels', 'scores'
            targets: List of target dicts with 'boxes', 'labels'
        """
        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes'].cpu()
            pred_labels = pred['labels'].cpu()
            pred_scores = pred['scores'].cpu()
            
            target_boxes = target['boxes'].cpu()
            target_labels = target['labels'].cpu()
            
            # Filter predictions by score threshold
            mask = pred_scores > self.score_threshold
            pred_boxes = pred_boxes[mask]
            pred_labels = pred_labels[mask]
            pred_scores = pred_scores[mask]
            
            self.predictions.append({
                'boxes': pred_boxes,
                'labels': pred_labels,
                'scores': pred_scores
            })
            
            self.targets.append({
                'boxes': target_boxes,
                'labels': target_labels
            })
            
    def compute(self) -> Dict[str, float]:
        """
        Compute detection metrics.
        
        Returns:
            Dictionary with mAP and per-class AP values
        """
        if not self.predictions:
            return {'mAP': 0.0}
            
        aps = []
        for c in range(self.num_classes):
            ap = self._compute_ap_for_class(c)
            if ap is not None:
                aps.append(ap)
                
        mAP = np.mean(aps) if aps else 0.0
        
        results = {'mAP': mAP}
        for c, ap in enumerate(aps):
            results[f'AP_class_{c}'] = ap
            
        return results
    
    def _compute_ap_for_class(self, class_id: int) -> Optional[float]:
        """Compute Average Precision for a specific class."""
        all_detections = []
        all_annotations = []
        
        for pred, target in zip(self.predictions, self.targets):
            # Get detections for this class
            mask = pred['labels'] == class_id
            boxes = pred['boxes'][mask]
            scores = pred['scores'][mask]
            
            all_detections.extend(
                [(box, score) for box, score in zip(boxes, scores)]
            )
            
            # Get ground truth for this class
            mask = target['labels'] == class_id
            boxes = target['boxes'][mask]
            all_annotations.extend([box for box in boxes])
            
        if not all_annotations:
            return None
            
        # Sort detections by score
        all_detections.sort(key=lambda x: x[1], reverse=True)
        
        tp = np.zeros(len(all_detections))
        fp = np.zeros(len(all_detections))
        
        for i, (pred_box, _) in enumerate(all_detections):
            best_iou = 0.0
            best_idx = -1
            
            for j, gt_box in enumerate(all_annotations):
                iou = self._compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
                    
            if best_iou >= self.iou_threshold:
                if not self.used_annotations.get(best_idx, False):
                    tp[i] = 1
                    self.used_annotations[best_idx] = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
                
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(all_annotations)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Append sentinel values
        precisions = np.concatenate([[1], precisions])
        recalls = np.concatenate([[0], recalls])
        
        # Compute average precision
        return np.trapz(precisions, recalls)
    
    @staticmethod
    def _compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def reset(self) -> None:
        """Reset metric state."""
        self.predictions = []
        self.targets = []
        self.used_annotations = {}
