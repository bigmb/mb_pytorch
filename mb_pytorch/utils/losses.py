import torch.nn.functional as F

__all__ = ['dice_loss', 'binary_cross_entropy', 'cross_entropy', 'mse_loss', 'l1_loss']

def dice_loss(input, target):
    """
    Dice loss function
    Formula: 1 - (2 * intersection + smooth) / (union + smooth)
    Input:
        input: torch tensor of shape (B, C, H, W)
        target: torch tensor of shape (B, C, H, W)
    Output:
        dice_loss: torch tensor of shape (1)
    """
    smooth = 1.
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))

def binary_cross_entropy(input, target):
    """
    Binary cross entropy loss function
    Formula: -sum(y * log(x) + (1 - y) * log(1 - x))
    Input:
        input: torch tensor of shape (B, C, H, W)
        target: torch tensor of shape (B, C, H, W)
    Output:
        binary_cross_entropy_loss: torch tensor of shape (1)
    """
    return F.binary_cross_entropy(input, target)

def cross_entropy(input, target):
    """
    Cross entropy loss function
    Formula: -sum(y * log(x))
    Input:
        input: torch tensor of shape (B, C, H, W)
        target: torch tensor of shape (B, C, H, W)
    Output:
        cross_entropy_loss: torch tensor of shape (1)
    """
    return F.cross_entropy(input, target)

def mse_loss(input, target):
    """
    Mean squared error loss function
    Formula: 1/N * sum((x - y)^2)
    Input:
        input: torch tensor of shape (B, C, H, W)
        target: torch tensor of shape (B, C, H, W)
    Output:
        mse_loss: torch tensor of shape (1)
    """
    return F.mse_loss(input, target)

def l1_loss(input, target):
    """
    L1 loss function
    Formula: |x - y|
    Input:
        input: torch tensor of shape (B, C, H, W)
        target: torch tensor of shape (B, C, H, W)
    Output:
        l1_loss: torch tensor of shape (1)
    """
    return F.l1_loss(input, target)

def hinge_loss(input, target):
    """
    Hinge loss function
    Formula: max(0, 1 - y * x)
    Input:
        input: torch tensor of shape (B, C, H, W)
        target: torch tensor of shape (B, C, H, W)
    Output:
        hinge_loss: torch tensor of shape (1)
    """
    return F.hinge_embedding_loss(input, target)

def l2_loss(input, target):
    """
    L2 loss function
    Formula: (x - y)^2
    Input:
        input: torch tensor of shape (B, C, H, W)
        target: torch tensor of shape (B, C, H, W)
    Output:
        l2_loss: torch tensor of shape (1)
    """
    return F.mse_loss(input, target)

