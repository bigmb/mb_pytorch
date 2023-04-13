from torchsummary import summary
import onnx
from onnx2pytorch import ConvertModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

__all__ = ['get_model_summary','onnx2torch','overwrite_layer_weights','feature_extractor']


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



def feature_extractor(model, layer_name):
    """
    Function to get the feature extractor from the model.
    Args:
        model (torch.nn.Module): PyTorch model
        layer_name (str): Name of the layer to be used as feature extractor
    Returns:
        feature_extractor (torch.nn.Sequential): Feature extractor from the model
    """
    # Get the named module from the model
    module = dict(model.named_modules())[layer_name]
    # Get the position of the named module in the list of all modules
    layer_loc = list(model.modules()).index(module)
    # Extract the feature extractor from the model
    feature_extractor = torch.nn.Sequential(*list(model.children())[:layer_loc])
    return feature_extractor


def feature_view(data,model,layer_names:list) -> None:
    """
    Function to view image from the feature extractor
        data: Image data
        model: PyTorch model
        layer_names (list): Name of the layer to be used as feature extractor
    Output:
        feature_view: Feature view from the model
    """
    out_list = []
    for j in data:
        for i in range(len(layer_names)):
            layer = layer_names[i]
            features = feature_extractor(model,layer)
            #features = getattr(model,layer)
            output = features(j)
            output = output.squeeze(0).detach().numpy()
            out_list.append(output)
        fig, axs = plt.subplots(1, len(layer_names), figsize=(12, 12))
        for i in range(len(layer_names)):
            if len(out_list[i].shape) > 2:
                out_map = np.sum(out_list[i], axis=0)
            else:
                out_map = out_list[i]
            sns.heatmap(out_map, ax=axs[i])
            axs[i].set_title(f'{layer_names[i]} : {out_list[i].shape}' )
    return None
        