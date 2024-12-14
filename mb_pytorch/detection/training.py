from typing import Optional, Dict, Any, List, Tuple
import torch
import numpy as np
from tqdm import tqdm
from ..training.base_trainer import BaseTrainer
from ..utils.viewer import plot_to_image
from mb.plt.utils import dynamic_plt

__all__ = ['DetectionTrainer', 'DetectionLoop']

class DetectionTrainer(BaseTrainer):
    """Trainer class specifically for object detection models."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        scheduler: Optional[Any] = None,
        writer: Optional[Any] = None,
        logger: Optional[Any] = None,
        gradcam: Optional[Any] = None,
        gradcam_rgb: bool = False,
        device: str = 'cpu'
    ):
        """
        Initialize the detection trainer.
        
        Args:
            config: Configuration dictionary
            scheduler: Optional learning rate scheduler
            writer: Optional tensorboard writer
            logger: Optional logger instance
            gradcam: Optional gradcam layers to visualize
            gradcam_rgb: Whether to use RGB for gradcam
            device: Device to run training on
        """
        super().__init__(config, scheduler, writer, logger, device)
        self.gradcam = gradcam
        self.gradcam_rgb = gradcam_rgb
        self.bbox_threshold = self.config['model']['model_meta_data']['model_bbox_threshold']
        
    def _prepare_batch(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Prepare a batch of data for training/validation.
        
        Args:
            batch: Dictionary containing images, bboxes, and labels
            
        Returns:
            Tuple of (images, targets) prepared for the model
        """
        images, bbox, labels = batch.values()
        images = [image.to(self.device) for image in images]
        bbox = [b.to(self.device) for b in bbox]
        bbox = [b.view(-1, 4) if b.dim() == 1 else b for b in bbox]
        labels = [label.to(self.device) for label in labels]
        
        targets = [
            {'boxes': b, 'labels': label} 
            for b, label in zip(bbox, labels)
        ]
        
        return images, targets
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        if self.logger:
            self.logger.info('Training Started')
            
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training", leave=False)):
            images, targets = self._prepare_batch(batch)
            
            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            losses.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += losses.item()
            
            if self.logger:
                self.logger.info(f'Epoch {epoch+1} - Batch {batch_idx+1} - Train Loss: {losses.item()}')
        
        return total_loss / len(self.train_loader)
    
    def validate_epoch(self, epoch: int) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        val_predictions = {
            'bbox': [], 'labels': [], 'scores': [],
            'targets_labels': [], 'targets_bbox': []
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation", leave=False)):
                images, targets = self._prepare_batch(batch)
                loss_dict = self.model(images, targets)
                
                if len(loss_dict) > 0:
                    self._process_predictions(loss_dict, targets, val_predictions)
                    
                    losses = sum(loss for loss in loss_dict.values())
                    total_loss += losses.item() * len(images)
                    
                    if self.logger:
                        self.logger.info(f'Epoch {epoch+1} - Batch {batch_idx+1} - Val Loss: {losses.item()}')
                
                # Visualize predictions if writer is available
                if self.writer is not None and len(images) > 0:
                    self._visualize_predictions(images, targets, val_predictions, epoch)
        
        return total_loss / len(self.val_loader.dataset)
    
    def _process_predictions(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
        val_predictions: Dict[str, List]
    ) -> None:
        """Process and store model predictions."""
        for pred, target in zip(predictions, targets):
            if len(pred['boxes']) > 0 and 'scores' in pred:
                for j, score in enumerate(pred['scores']):
                    if score > self.bbox_threshold:
                        val_predictions['bbox'].append(pred['boxes'][j])
                        val_predictions['labels'].append(pred['labels'][j])
                        val_predictions['scores'].append(score)
                        if j < 1:  # Store first target for each prediction
                            val_predictions['targets_labels'].append(target['labels'])
                            val_predictions['targets_bbox'].append(target['boxes'])
    
    def _visualize_predictions(
        self,
        images: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        val_predictions: Dict[str, List],
        epoch: int
    ) -> None:
        """Visualize predictions using tensorboard."""
        img_list = [np.array(img.to('cpu')) for img in images]
        labels_list = [
            str(list(np.array((target['labels'].to('cpu'))))[0]) 
            for target in targets
        ]
        
        fig = dynamic_plt(
            img_list,
            labels=labels_list,
            bboxes=val_predictions['targets_bbox'],
            return_fig=True
        )
        self.writer.add_image('grid', plot_to_image(fig), global_step=epoch)

def DetectionLoop(
    k_yaml: dict,
    scheduler: Optional[object] = None,
    writer: Optional[object] = None,
    logger: Optional[object] = None,
    gradcam: Optional[object] = None,
    gradcam_rgb: bool = False,
    device: str = 'cpu'
) -> None:
    """
    Main training function for object detection.
    
    Args:
        k_yaml: Configuration dictionary
        scheduler: Optional scheduler
        writer: Optional tensorboard writer
        logger: Optional logger
        gradcam: Optional gradcam layers
        gradcam_rgb: Whether to use RGB for gradcam
        device: Device to use for training
    """
    trainer = DetectionTrainer(
        k_yaml.data_dict,
        scheduler=scheduler,
        writer=writer,
        logger=logger,
        gradcam=gradcam,
        gradcam_rgb=gradcam_rgb,
        device=device
    )
    trainer.train()
