from typing import Optional, Dict, Any, Tuple
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import os
from tqdm import tqdm
from mb_utils.src.logging import logger
from ..models.modelloader import ModelLoader
from ..utils import losses as loss_fn
import inspect

__all__ = ['BaseTrainer']

class BaseTrainer:
    """Base trainer class that handles common training functionality."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        train_loader: DataLoader = None,
        val_loader: DataLoader = None,
        writer: Optional[Any] = None,
        logger: Optional[Any] = None,
        device: str = 'cpu'
    ):
        """
        Initialize the trainer with configuration and optional components.
        
        Args:
            config: Configuration dictionary containing model and data settings
            writer: Optional tensorboard writer
            logger: Optional logger instance
            device: Device to run training on ('cpu' or 'cuda')
        """
        self.config = config
        self.writer = writer
        self.logger = logger
        self.device = self._setup_device(device)
        self.model_params = self.config['model']

        # Initialize training components
        self.model = self._setup_model()
        self.train_loader = train_loader
        self.val_loader = val_loader
        # self.train_loader, self.val_loader = self._setup_data()
        self.loss_fn, self.optimizer = self._setup_training()
        
        self.best_val_loss = float('inf')
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup training device."""
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if self.logger:
                self.logger.info(f'Using CUDA: {torch.cuda.get_device_name(0)}')
            return torch.device('cuda')
        return torch.device('cpu')
    
    def _setup_model(self) -> torch.nn.Module:
        """Setup and return the model."""
        if self.logger:
            self.logger.info('Loading model...')
        model_loader = ModelLoader(self.config['model'])
        model = model_loader.get_model()
        model.to(self.device)
        return model
    
    # def _setup_data(self) -> Tuple[DataLoader, DataLoader]:
    #     """Setup and return data loaders."""
    #     if self.logger:
    #         self.logger.info('Setting up data loaders...')
    #     train_loader, val_loader, _, _ = self.config.data_load()
    #     return train_loader, val_loader
    
    def _setup_train_helper(self) -> Tuple[Any, Optimizer]:
        
        if self.model_params['model_optimizer'] is not None:
            optimizer = getattr(torch.optim,self.model_params['model_optimizer'])
            temp_str = self.model_params['model_optimizer']
            optimizer_dict = self.model_params['model_train_parameters'][temp_str]
        else:
            optimizer_dict = self.model_params['model_train_parameters']['Adam']
            optimizer = getattr(torch.optim,'Adam')
        
        if self.model_params['model_scheduler'] is not None:
            scheduler_dict = self.model_params['model_train_parameters'][self.model_params['model_scheduler']]
            #print(scheduler_dict)
            #print(data['model_scheduler'])
            scheduler = getattr(torch.optim.lr_scheduler,self.model_params['model_scheduler'])
            #print(scheduler_attr)
            #scheduler = scheduler_attr(optimizer,**scheduler_dict)
        else:
            scheduler = None
            scheduler_dict = None
        
        loss = None
        for _,k in enumerate(inspect.getmembers(loss_fn)):
            if self.model_params['model_loss']==k:
                loss_attr = getattr(loss_fn,self.model_params['model_loss'])
        if loss==None:
            loss_attr = getattr(torch.nn.functional,self.model_params['model_loss']) ## if loss is not in loss_fn, it is in torch.nn

        return loss_attr, optimizer, optimizer_dict, scheduler, scheduler_dict

    def _setup_training(self) -> Tuple[Any, Optimizer]:
        """Setup loss function and optimizer."""
        if self.logger:
            self.logger.info('Setting up training components...')
        loss_fn, optimizer_cls, optimizer_params, scheduler_cls, scheduler_params = self._setup_train_helper()
        optimizer = optimizer_cls(self.model.parameters(), **optimizer_params)
        
        if scheduler_cls is not None:
            self.scheduler = scheduler_cls(optimizer, **scheduler_params)
            
        if self.logger:
            self.logger.info(f'Loss function: {loss_fn.__name__}')
            self.logger.info(f'Optimizer: {optimizer.__class__.__name__}')
            self.logger.info(f'Scheduler: {self.scheduler.__class__.__name__ if self.scheduler else None}')
            
        return loss_fn, optimizer
    
    def save_model(self, val_loss: float, epoch: int) -> None:
        """Save model if validation loss improves."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            save_path = os.path.join(
                os.path.dirname(self.config['data']['file']['root']), 
                'best_model.pth'
            )
            torch.save(self.model.state_dict(), save_path)
            if self.logger:
                self.logger.info(f'Epoch {epoch + 1} - Best Model Saved (Val Loss: {self.best_val_loss:.4f})')
    
    def log_metrics(self, train_loss: float, val_loss: float, epoch: int) -> None:
        """Log training metrics to tensorboard if writer is available."""
        if self.writer is not None:
            self.writer.add_scalar('Loss/train', train_loss, global_step=epoch)
            self.writer.add_scalar('Loss/val', val_loss, global_step=epoch)
            self.writer.add_scalar(
                'Learning_rate', 
                self.optimizer.param_groups[0]['lr'], 
                global_step=epoch
            )
            
            # Log model parameter histograms
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(name, param, global_step=epoch)
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        raise NotImplementedError("Subclasses must implement train_epoch")
    
    def validate_epoch(self, epoch: int) -> float:
        """
        Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average validation loss for the epoch
        """
        raise NotImplementedError("Subclasses must implement validate_epoch")
    
    def train(self) -> None:
        """Main training loop."""
        if self.logger:
            self.logger.info('Starting training...')
            
        num_epochs = self.config['model']['model_epochs']
        
        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            # Training phase
            train_loss = self.train_epoch(epoch)
            if self.logger:
                self.logger.info(f'Epoch {epoch + 1} - Train Loss: {train_loss:.4f}')
            
            # Validation phase
            val_loss = self.validate_epoch(epoch)
            if self.logger:
                self.logger.info(f'Epoch {epoch + 1} - Val Loss: {val_loss:.4f}')
            
            # Log metrics and save model
            self.log_metrics(train_loss, val_loss, epoch)
            self.save_model(val_loss, epoch)
