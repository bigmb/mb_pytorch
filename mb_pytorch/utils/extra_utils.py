from torchsummary import summary
import onnx
from onnx2pytorch import ConvertModel
import torch

__all__ = ['get_model_summary','onnx2torch','overwrite_layer_weights']


def get_model_summary(model, input_size):
    """Prints the model summary.
    Input:
        model: PyTorch model
        input_size: input size of the model
    """
    summary(model, input_size=input_size)
    
def onnx2torch(model):
    """
    Function to convert onnx model to torch model
    Input:
        model: onnx model
    Output:
        torch model
    """
    onnx_m = onnx.load(model)
    torch_m = ConvertModel(onnx_m)
    return torch_m


def overwrite_layer_weights(model, layer_index, new_weights,logger=None):
    """
    Overwrites the weights of a specified layer of a PyTorch model with the given weights.
    Args:
    - model: A PyTorch model.
    - layer_index: The index of the layer whose weights should be overwritten.
    - new_weights: A tensor containing the new weights to be used for the specified layer.
    """

    layer_name = list(model.named_modules())[layer_index][0]
    layer = getattr(model, layer_name)
    if logger:
        logger.info("Overwriting the weights of layer {} with the given weights.".format(layer_name))
    if isinstance(layer, torch.nn.Conv2d):
        layer.weight.data = new_weights
    if isinstance(layer, torch.nn.Linear):
        layer.weight.data = new_weights
    else:
        raise ValueError("The specified layer is not a convolutional layer or linear layer.")

