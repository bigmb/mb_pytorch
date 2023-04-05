from ..dataloader.loader import DataLoader
from mb_utils.src.logging import logger
import torch
from ..utils import losses as loss_fn
import inspect

__all__ = ['train_helper']

def train_helper(data):
    """
    Function to get optimizers, learning rate,scheduler, loss
    """

    if optimizer is not None:
        optimizer = getattr(torch.optim,data['model_optimizer'])
        optimizer_dict = data['model_train_parameters']['model_optimizer']
    else:
        optimizer_dict = data['model_train_parameters']['Adam']
        optimizer = getattr(torch.optim,'Adam')
    
    if data['model_scheduler'] is not None:
        scheduler_dict = data['model_train_parameters']['model_scheduler']
        scheduler = getattr(torch.optim.lr_scheduler,data['model_scheduler'])(optimizer,**scheduler_dict)
    else:
        scheduler = None
        scheduler_dict = None
    
    loss = None
    for _,k in enumerate(inspect.getmembers(loss_fn)):
        if data['model_loss']==k:
            loss = getattr(loss_fn,data['model_loss'])
    if loss==None:
        loss = getattr(torch.nn,data['model_loss']) ## if loss is not in loss_fn, it is in torch.nn

    return loss, optimizer, optimizer_dict, scheduler, scheduler_dict
        
    