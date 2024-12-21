from typing import Optional, Dict, Any, List, Tuple
import torch
import numpy as np
from tqdm import tqdm
from ..training.base_trainer import BaseTrainer
from ..utils.viewer import plot_to_image
from ..utils.compiler import TorchScriptUtils
from mb.plt.utils import dynamic_plt
from collections import OrderedDict
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torch.utils.tensorboard import SummaryWriter
import os

__all__ = ['DetectionTrainer', 'DetectionLoop']

class DetectionTrainer(BaseTrainer):
    """Trainer class specifically for object detection models."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        train_loader: torch.utils.data.DataLoader = None,
        val_loader: torch.utils.data.DataLoader = None,
        writer: Optional[Any] = None,
        logger: Optional[Any] = None,
        gradcam: Optional[Any] = None,
        gradcam_rgb: bool = False,
        device: str = 'cpu',
        use_all_cpu_cores: bool = False,
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
            use_all_cpu_cores: Whether to use all CPU cores for data loading. (2 cpu cores less than max. default)
        """
        super().__init__(config, train_loader,val_loader, writer, logger, device)
        self.gradcam = gradcam
        self.gradcam_rgb = gradcam_rgb
        if use_all_cpu_cores:
            TorchScriptUtils.set_to_max_cores()
        self.bbox_threshold = self.config['model']['model_meta_data']['model_bbox_threshold']
        if writer:
            self.writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(self.config['data']['file']['root']), 'logs'))

        
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
        # labels = list(label.to('cpu') for label in labels)  
        labels = [torch.tensor([label.to('cpu').tolist()]).to(self.device) for label in labels]
        
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
            loss_dict = self.model(images,targets)
            losses = sum(loss for loss in loss_dict.values())
            
            losses.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += losses.item()
            
            if batch_idx % 10 == 0 and self.logger:
                self.logger.info(f"Epoch {epoch + 1}, Batch {batch_idx}: Loss={losses.item():.4f}")
            else:
                print(f"Epoch {epoch + 1}, Batch {batch_idx}: Loss={losses.item():.4f}")

        return total_loss / len(self.train_loader)
    
    def _eval_forward_new(self, model, images, targets):
        """Simplified evaluation logic."""
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = model.transform(images, targets)

        # Simplified feature extraction
        features = model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # RPN proposals
        proposals, proposal_losses = model.rpn(images, features, targets)

        # ROI heads
        detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
        detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        # Combine losses
        losses = {**detector_losses, **proposal_losses}
        return losses, detections
    
    def _eval_forward(self,model, images, targets):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                It returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        model.eval()

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = model.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                    )

        features = model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        model.rpn.training=True
        #model.roi_heads.training=True


        #####proposals, proposal_losses = model.rpn(images, features, targets)
        features_rpn = list(features.values())
        objectness, pred_bbox_deltas = model.rpn.head(features_rpn)
        anchors = model.rpn.anchor_generator(images, features_rpn)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        proposals, scores = model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        proposal_losses = {}
        assert targets is not None
        labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(anchors, targets)
        regression_targets = model.rpn.box_coder.encode(matched_gt_boxes, anchors)
        loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(
            objectness, pred_bbox_deltas, labels, regression_targets
        )
        proposal_losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }

        #####detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
        image_shapes = images.image_sizes
        proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(proposals, targets)
        box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
        box_features = model.roi_heads.box_head(box_features)
        class_logits, box_regression = model.roi_heads.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        detector_losses = {}
        loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
        detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        boxes, scores, labels = model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )
        detections = result
        detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]
        model.rpn.training=False
        model.roi_heads.training=False
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses, detections
    
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
                # loss_dict = self.model(images) ##old code - doesnt give loss
                loss_dict, detections = self._eval_forward(self.model, images, targets)
                # loss_dict, detections = self._eval_forward_new(self.model, images, targets) #testing new function

                if len(loss_dict) > 0:
                    self._process_predictions(detections, targets, val_predictions)
                    
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

    def test_profile(self):
        """Profile the model inference."""
        import torch.profiler
        with torch.profiler.profile(
            activities=[
                        torch.profiler.ProfilerActivity.CPU, 
                        torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(os.path.dirname(self.config['data']['file']['root']), 'log_profiler'))) as p:
            for batch_idx, batch in enumerate(self.train_loader):
                images, targets = self._prepare_batch(batch)
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()  
                self.optimizer.step()
                p.step()
                if batch_idx > 10:
                    break


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
