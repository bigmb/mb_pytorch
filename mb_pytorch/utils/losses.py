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

def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1 (torch.Tensor): The first bounding box, of the form [x1, y1, x2, y2].
        box2 (torch.Tensor): The second bounding box, of the form [x1, y1, x2, y2].
    
    Returns:
        iou (float): The Intersection over Union of the two boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    iou = intersection / float(box1_area + box2_area - intersection)
    
    return iou


def compute_loss(outputs, targets):
    """
    Compute the loss for a FasterRCNN model with a ResNet backbone.
    
    Args:
        outputs (dict): A dictionary containing the model's outputs, including 'boxes', 'labels', and 'scores'.
        targets (dict): A dictionary containing the ground truth bounding boxes and labels.
    
    Returns:
        loss (torch.Tensor): The total loss.
    """
    # Extract the relevant outputs and targets
    predicted_boxes = outputs['boxes']
    predicted_labels = outputs['labels']
    
    gt_boxes = targets['boxes']
    gt_labels = targets['labels']
    
    # Compute the classification loss
    classification_loss = F.cross_entropy(predicted_labels, gt_labels)
    
    # Compute the bounding box regression loss
    num_predicted = predicted_boxes.size(0)
    num_ground_truth = gt_boxes.size(0)
    
    bbox_regression_loss = 0
    for i in range(num_predicted):
        # Find the best matching ground truth box
        best_iou = 0
        best_index = -1
        for j in range(num_ground_truth):
            iou = compute_iou(predicted_boxes[i], gt_boxes[j])
            if iou > best_iou:
                best_iou = iou
                best_index = j
        
        # Compute the smooth L1 loss for the best matching box
        if best_index >= 0:
            bbox_regression_loss += F.smooth_l1_loss(predicted_boxes[i], gt_boxes[best_index])
    
    bbox_regression_loss /= num_predicted
    
    # Compute the total loss
    total_loss = classification_loss + bbox_regression_loss
    
    return total_loss ##look into adding boundingbox + classification loss for validation