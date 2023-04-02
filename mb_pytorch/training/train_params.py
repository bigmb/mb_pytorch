from ..dataloader.loader import DataLoader
from mb_utils.src.logging import logger
import torch

__all__ = ['train_helper']

def train_helper(data):
    """
    Function to get optimizers, learning rate,scheduler, loss
    """
    loss = data['loss']
    optimizer = data['optimizer']
    learning_rate = data['learning_rate']
    scheduler = data['scheduler']
    


