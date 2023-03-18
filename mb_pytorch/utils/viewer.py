import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torchvision.transforms.functional as TF

__all__ = ['show_images', 'show_segmentation_masks', 'show_bounding_boxes', 'show_label_on_img','model_viewer']

def show_images(imgs, figsize=(12.0, 12.0)):
    """Displays a single image or list of images. 
    Args:
        imgs (Union[List[torch.Tensor], torch.Tensor]): A list of images
            of shape (3, H, W) or a single image of shape (3, H, W).
        figsize (Tuple[float, float]): size of figure to display.
    Returns:
        None
    """

    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), figsize=figsize, squeeze=False)
    for i, img in enumerate(imgs):
        #img = img.detach()
        #img = TF.to_pil_image(img)
        img = np.asarray(img)
        if img.shape[0]==3:
            img = np.transpose(img, (1, 2, 0))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

    return None

def show_segmentation_masks(imgs, masks, figsize=(12.0, 12.0)):
    """Displays a single image or list of images with segmentation masks.
    Args:
        imgs (Union[List[torch.Tensor], torch.Tensor]): A list of images
            of shape (3, H, W) or a single image of shape (3, H, W).
        masks (Union[List[torch.Tensor], torch.Tensor]): A list of masks
            of shape (1, H, W) or a single mask of shape (1, H, W).
        figsize (Tuple[float, float]): size of figure to display.
    Returns:
        None
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    if not isinstance(masks, list):
        masks = [masks]
    fig, axs = plt.subplots(ncols=len(imgs), figsize=figsize, squeeze=False)
    for i, (img, mask) in enumerate(zip(imgs, masks)):
        # img = img.detach()
        # img = TF.to_pil_image(img)
        # mask = mask.detach()
        # mask = TF.to_pil_image(mask)
        img = np.asarray(img)
        if img.shape[0]==3:
            img = np.transpose(img, (1, 2, 0))
        mask = np.asarray(mask)
        if mask.shape[0]==3:
            mask = np.transpose(mask, (1, 2, 0))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].imshow(np.asarray(mask), alpha=0.5)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

    return None

def show_bounding_boxes(imgs, boxes, figsize=(12.0, 12.0)):
    """Displays a single image or list of images with bounding boxes.
    Args:
        imgs (Union[List[torch.Tensor], torch.Tensor]): A list of images
            of shape (3, H, W) or a single image of shape (3, H, W).
        boxes (Union[List[torch.Tensor], torch.Tensor]): A list of boxes
            of shape (N, 4) or a single box of shape (N, 4).
        figsize (Tuple[float, float]): size of figure to display.
    Returns:
        None
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    if not isinstance(boxes, list):
        boxes = [boxes]
    fig, axs = plt.subplots(ncols=len(imgs), figsize=figsize, squeeze=False)
    for i, (img, box) in enumerate(zip(imgs, boxes)):
        # img = img.detach()
        # img = TF.to_pil_image(img)
        # box = box.detach()
        img = np.asarray(img)
        if img.shape[0]==3:
            img = np.transpose(img, (1, 2, 0))
        box = np.asarray(box)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].imshow(draw_bounding_boxes(img, box))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

    return None

def show_label_on_img(imgs,labels,figsize=(12.0,12.0)):
    """Displays a single image or list of images with labels.
    Args:
        imgs (Union[List[torch.Tensor], torch.Tensor]): A list of images
            of shape (3, H, W) or a single image of shape (3, H, W).
        labels (str): label to be displayed on image.
        figsize (Tuple[float, float]): size of figure to display.
    Returns:
        None
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    if not isinstance(labels, list):
        labels = [labels]
    fig, axs = plt.subplots(ncols=len(imgs), figsize=figsize, squeeze=False)
    for i, (img, label) in enumerate(zip(imgs, labels)):
        # img = img.detach()
        # img = TF.to_pil_image(img)
        img = np.asarray(img)
        if img.shape[0]==3:
            img = np.transpose(img, (1, 2, 0))
        img = img.astype('float32')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.putText(img, str(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1)
        axs[0, i].imshow(np.asarray(img))
    plt.show()

    return None

def model_viewer(model,input_shape = (1,3,128,128), location='model',view=False):
    """
    Function to visualize the model architecture
    Uses torchviz to visualize the model
    Input:
        model: Model to be visualized
        input_shape: Input shape of the model (batch,channels,height,width)
        location: name and location to save the model image. Default is current directory
        view: Boolean to view the model image or not
    Output:
        None
    """
    from torchviz import make_dot
    x = torch.randn(input_shape)
    y = model(x)
    dot= make_dot(y.mean(), params=dict(model.named_parameters()),show_attrs=True)
    dot.format = 'png'
    dot.render(location,view=view)
    return None
    