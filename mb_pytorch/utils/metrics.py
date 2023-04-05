import numpy as np
from scipy import spatial

__all__ = ['dice_coefficient_image','numeric_score']

def dice_coefficient_image(pred, target,empty_score=1.0):
    """
    Calculate the dice coefficient between two images
    Formula: 2 * intersection / (pred + target)
    Input:
        pred: numpy array of shape (H, W)
        target: numpy array of shape (H, W)
        empty_score: float
    Output:
        dice_coefficient: float
    """
    
    im1 = np.asarray(pred).astype(np.bool)
    im2 = np.asarray(target).astype(np.bool)
    
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    
    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score
    
    intersection = np.logical_and(im1, im2)    
    return 2. * intersection.sum() / im_sum

def numeric_score(pred, target):
    """
    Compute the numeric score between two images like FP, FN, TP, TN
    Formula: FP, FN, TP, TN
    Input:
        pred: numpy array of shape (H, W)
        target: numpy array of shape (H, W)
    Output:
        numeric_score: list of int
    """
    
    FP = np.logical_and(pred, np.logical_not(target)).sum()
    