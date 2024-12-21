# MB PyTorch

A comprehensive PyTorch-based machine learning library providing implementations for object detection, classification, segmentation, and meta-learning tasks.

[![Close inactive issues](https://github.com/bigmb/mb_pytorch/actions/workflows/stale.yml/badge.svg)](https://github.com/bigmb/mb_pytorch/actions/workflows/stale.yml)
[![Downloads](https://static.pepy.tech/personalized-badge/mb-pytorch?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads)](https://pepy.tech/project/mb-pytorch)

## Features

- **Object Detection**: Implementation of object detection models and training pipelines
- **Classification**: Tools and models for image classification tasks
- **Segmentation**: Semantic segmentation models and utilities
- **Meta Learning**: Support for meta-learning approaches
- **Flexible Data Loading**: Configurable data loading with YAML-based configuration
- **Visualization**: Integration with Gradio and TensorBoard for model visualization

## Installation

Requirements:
- Python >= 3.8
- PyTorch
- torchvision

```bash
pip install mb_pytorch
```

Additional dependencies:
```
torchsummary
numpy
matplotlib
mb_pandas
mb_utils
onnx
onnx2pytorch
seaborn
tqdm
pytorch_grad_cam
pillow
opencv-python
mb_base
```

## Usage

### Object Detection Example

```python
from mb_pytorch.utils.yaml_reader import YAMLReader
from mb_pytorch.dataloader.loader import BaseDataset, TorchDataLoader
from mb_pytorch.detection.training import DetectionTrainer

# Load configuration
yaml_read = YAMLReader('scripts/detection/object_detection.yaml')
yaml_data = yaml_read.read()

# Create datasets
train_dataset = BaseDataset(
    data_config=yaml_data['data']['file'],
    task_type=yaml_data['model']['model_type'],
    transform=yaml_data['transformation'],
    is_train=True
)

val_dataset = BaseDataset(
    data_config=yaml_data['data']['file'],
    task_type=yaml_data['model']['model_type'],
    transform=yaml_data['transformation'],
    is_train=False
)

# Create dataloaders
train_dataloader = TorchDataLoader(
    dataset=train_dataset,
    batch_size=yaml_data['train_params']['batch_size'],
    shuffle=yaml_data['train_params']['shuffle'],
    num_workers=yaml_data['train_params']['num_workers']
)

val_dataloader = TorchDataLoader(
    dataset=val_dataset,
    batch_size=yaml_data['val_params']['batch_size'],
    shuffle=yaml_data['val_params']['shuffle'],
    num_workers=yaml_data['val_params']['num_workers']
)

# Initialize trainer and train
trainer = DetectionTrainer(
    yaml_data,
    train_dataloader,
    val_dataloader,
    device='cpu',
    use_all_cpu_cores=True
)
trainer.train()
```

### Data Loading
The library provides flexible data loading capabilities through YAML configuration:

```python
from mb_pytorch.dataloader.loader import DataLoader
from mb_utils.src.logging import logger

loader = DataLoader('./scripts/loader_y.yaml')
out1, out2, o1, o2 = loader.data_load(logger=logger)
```

## Project Structure

```
mb_pytorch/
├── classification/    # Classification models and utilities
├── dataloader/       # Data loading utilities
├── detection/        # Object detection implementations
├── metalearning/     # Meta-learning implementations
├── metrics/         # Evaluation metrics
├── models/          # Base model implementations
├── segmentation/    # Segmentation models
├── training/        # Training utilities
└── utils/           # Helper utilities
```

## Visualization

The library supports multiple visualization options:
- **Gradio**: Interactive model visualization and testing
- **TensorBoard**: Training metrics and model performance visualization

## Examples

Check the `examples/` directory for Jupyter notebooks demonstrating various use cases:
- Object Detection
- Basic Data Loading
- Embedding Data Loading
- Model Visualization
- Segmentation Models
