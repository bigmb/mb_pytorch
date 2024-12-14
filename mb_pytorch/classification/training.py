from typing import Optional, Dict, Any, List, Tuple
import torch
import numpy as np
from tqdm import tqdm
from ..training.base_trainer import BaseTrainer
from ..utils.viewer import plot_to_image
from mb.plt.utils import dynamic_plt

__all__ = ['ClassificationTrainer', 'ClassificationLoop']

class ClassificationTrainer(BaseTrainer):
    """Trainer class specifically for classification models."""
    
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
        Initialize the classification trainer.
        
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
        
    def _prepare_batch(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare a batch of data for training/validation.
        
        Args:
            batch: Dictionary containing images and labels
            
        Returns:
            Tuple of (images, labels) prepared for the model
        """
        images, labels = batch['images'], batch['labels']
        return images.to(self.device), labels.to(self.device)
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        if self.logger:
            self.logger.info('Training Started')
            
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training", leave=False)):
            images, labels = self._prepare_batch(batch)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += loss.item()
            
            if self.logger:
                self.logger.info(f'Epoch {epoch+1} - Batch {batch_idx+1} - Train Loss: {loss.item()}')
        
        return total_loss / len(self.train_loader)
    
    def validate_epoch(self, epoch: int) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation", leave=False)):
                images, labels = self._prepare_batch(batch)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                if self.logger:
                    self.logger.info(f'Epoch {epoch+1} - Batch {batch_idx+1} - Val Loss: {loss.item()}')
                
                # Visualize predictions if writer is available
                if self.writer is not None and batch_idx == 0:  # Visualize first batch
                    self._visualize_predictions(images, labels, predicted, epoch)
        
        accuracy = 100. * correct / total
        if self.writer is not None:
            self.writer.add_scalar('Accuracy/val', accuracy, global_step=epoch)
        
        return total_loss / len(self.val_loader.dataset)
    
    def _visualize_predictions(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        epoch: int
    ) -> None:
        """Visualize predictions using tensorboard."""
        img_list = [np.array(img.cpu()) for img in images[:8]]  # Visualize up to 8 images
        labels_list = [f"True: {l.item()}, Pred: {p.item()}" 
                      for l, p in zip(labels[:8], predictions[:8])]
        
        fig = dynamic_plt(
            img_list,
            labels=labels_list,
            return_fig=True
        )
        self.writer.add_image('predictions', plot_to_image(fig), global_step=epoch)

def ClassificationLoop(
    k_yaml: dict,
    scheduler: Optional[object] = None,
    writer: Optional[object] = None,
    logger: Optional[object] = None,
    gradcam: Optional[object] = None,
    gradcam_rgb: bool = False,
    device: str = 'cpu'
) -> None:
    """
    Main training function for classification.
    
    Args:
        k_yaml: Configuration dictionary
        scheduler: Optional scheduler
        writer: Optional tensorboard writer
        logger: Optional logger
        gradcam: Optional gradcam layers
        gradcam_rgb: Whether to use RGB for gradcam
        device: Device to use for training
    """
    trainer = ClassificationTrainer(
        k_yaml.data_dict,
        scheduler=scheduler,
        writer=writer,
        logger=logger,
        gradcam=gradcam,
        gradcam_rgb=gradcam_rgb,
        device=device
    )
    trainer.train()
