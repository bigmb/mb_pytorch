"""Utilities for visualization of images, masks, and model outputs."""

from typing import List, Optional, Union, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import io
from PIL import Image
import seaborn as sns
from pytorch_grad_cam import GradCAM

__all__ = [
    'ImageVisualizer',
    'GradCAMVisualizer',
    'TensorboardVisualizer'
]

class ImageVisualizer:
    """Visualization utilities for images, masks, and bounding boxes."""
    
    @staticmethod
    def show_images(
        images: Union[List[torch.Tensor], torch.Tensor],
        labels: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (12.0, 12.0)
    ) -> None:
        """
        Display images with optional labels.
        
        Args:
            images: List of images or single image tensor (3, H, W)
            labels: Optional list of labels to display
            figsize: Figure size
        """
        if not isinstance(images, list):
            images = [images]
            
        fig, axs = plt.subplots(ncols=len(images), figsize=figsize, squeeze=False)
        
        for i, img in enumerate(images):
            img_array = np.asarray(img)
            if img_array.shape[0] == 3:
                img_array = np.transpose(img_array, (1, 2, 0))
                
            axs[0, i].imshow(img_array)
            if labels and i < len(labels):
                axs[0, i].set_title(labels[i])
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            
        plt.show()
    
    @staticmethod
    def show_segmentation_masks(
        images: Union[List[torch.Tensor], torch.Tensor],
        masks: Union[List[torch.Tensor], torch.Tensor],
        alpha: float = 0.5,
        figsize: Tuple[float, float] = (12.0, 12.0)
    ) -> None:
        """
        Display images with segmentation masks.
        
        Args:
            images: List of images or single image tensor (3, H, W)
            masks: List of masks or single mask tensor (1, H, W)
            alpha: Mask transparency
            figsize: Figure size
        """
        if not isinstance(images, list):
            images = [images]
        if not isinstance(masks, list):
            masks = [masks]
            
        fig, axs = plt.subplots(ncols=len(images), figsize=figsize, squeeze=False)
        
        for i, (img, mask) in enumerate(zip(images, masks)):
            img_array = np.asarray(img)
            if img_array.shape[0] == 3:
                img_array = np.transpose(img_array, (1, 2, 0))
                
            mask_array = np.asarray(mask)
            if mask_array.shape[0] == 1:
                mask_array = np.transpose(mask_array, (1, 2, 0))
                
            axs[0, i].imshow(img_array)
            axs[0, i].imshow(mask_array, alpha=alpha)
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            
        plt.show()
    
    @staticmethod
    def show_bounding_boxes(
        images: Union[List[torch.Tensor], torch.Tensor],
        boxes: Union[List[torch.Tensor], torch.Tensor],
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (12.0, 12.0)
    ) -> None:
        """
        Display images with bounding boxes.
        
        Args:
            images: List of images or single image tensor (3, H, W)
            boxes: List of boxes or single box tensor (N, 4)
            labels: Optional list of labels for each box
            colors: Optional list of colors for each box
            figsize: Figure size
        """
        if not isinstance(images, list):
            images = [images]
        if not isinstance(boxes, list):
            boxes = [boxes]
            
        if colors is None:
            colors = ['red'] * len(boxes)
            
        fig, axs = plt.subplots(ncols=len(images), figsize=figsize, squeeze=False)
        
        for i, (img, box) in enumerate(zip(images, boxes)):
            img_array = np.asarray(img)
            if img_array.shape[0] == 3:
                img_array = np.transpose(img_array, (1, 2, 0))
                
            axs[0, i].imshow(img_array)
            
            box_array = box.numpy() if isinstance(box, torch.Tensor) else box
            for j, bbox in enumerate(box_array):
                x1, y1, x2, y2 = bbox
                rect = plt.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    fill=False,
                    edgecolor=colors[j % len(colors)],
                    linewidth=2
                )
                axs[0, i].add_patch(rect)
                
                if labels and j < len(labels):
                    axs[0, i].text(
                        x1, y1-5,
                        labels[j],
                        color=colors[j % len(colors)],
                        fontsize=10
                    )
                    
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            
        plt.show()


class GradCAMVisualizer:
    """Visualization utilities for GradCAM."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: List[torch.nn.Module],
        use_cuda: bool = False
    ):
        """
        Initialize GradCAM visualizer.
        
        Args:
            model: Model to visualize
            target_layers: List of target layers for GradCAM
            use_cuda: Whether to use CUDA
        """
        self.model = model
        self.target_layers = target_layers
        self.use_cuda = use_cuda
        
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_category: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate CAM for input tensor.
        
        Args:
            input_tensor: Input image tensor
            target_category: Optional target category
            
        Returns:
            CAM visualization array
        """
        with GradCAM(
            model=self.model,
            target_layers=self.target_layers,
            use_cuda=self.use_cuda
        ) as cam:
            grayscale_cam = cam(
                input_tensor=input_tensor,
                target_category=target_category
            )[0, :]
            return grayscale_cam
    
    def visualize_cam(
        self,
        image: np.ndarray,
        cam_mask: np.ndarray,
        use_rgb: bool = True,
        image_weight: float = 0.4
    ) -> np.ndarray:
        """
        Visualize CAM on image.
        
        Args:
            image: Original image
            cam_mask: CAM mask
            use_rgb: Whether to use RGB
            image_weight: Weight of original image in visualization
            
        Returns:
            Visualization array
        """
        cam_mask = np.float32(cam_mask) / 255
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_mask),
            cv2.COLORMAP_JET
        )
        
        if use_rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255
        
        if image_weight < 0 or image_weight > 1:
            raise ValueError(
                f"image_weight should be in [0, 1], got: {image_weight}"
            )
        
        if use_rgb:
            cam = (1 - image_weight) * heatmap[..., ::-1] + image_weight * image
        else:
            cam = (1 - image_weight) * heatmap[..., 0] + image_weight * image
            
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)


class TensorboardVisualizer:
    """Utilities for Tensorboard visualization."""
    
    @staticmethod
    def figure_to_image(figure: plt.Figure) -> torch.Tensor:
        """
        Convert matplotlib figure to tensor for Tensorboard.
        
        Args:
            figure: Matplotlib figure
            
        Returns:
            Image tensor
        """
        buf = io.BytesIO()
        figure.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        return torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
    
    @staticmethod
    def create_image_grid(
        images: torch.Tensor,
        labels: torch.Tensor,
        predictions: Optional[torch.Tensor] = None,
        num_images: int = 16,
        cols: int = 4
    ) -> plt.Figure:
        """
        Create image grid for Tensorboard.
        
        Args:
            images: Batch of images
            labels: True labels
            predictions: Optional predicted labels
            num_images: Number of images to display
            cols: Number of columns in grid
            
        Returns:
            Matplotlib figure
        """
        rows = num_images // cols
        figure = plt.figure(figsize=(2*cols, 2*rows))
        
        for i in range(num_images):
            plt.subplot(rows, cols, i + 1)
            plt.axis('off')
            
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            plt.imshow(img)
            
            if predictions is not None:
                color = 'green' if predictions[i] == labels[i] else 'red'
                plt.title(f'Pred: {predictions[i]}\nTrue: {labels[i]}', color=color)
            else:
                plt.title(f'Label: {labels[i]}')
                
        plt.tight_layout()
        return figure
