from mb_utils.src.logging import logger
from torch import nn
import torch
from torch.nn import functional as F
import torchvision
import torchvision.models as models
import numpy as np
import importlib
from tqdm import tqdm

__all__ = ['ModelLoader','LayerExtractor']


def get_custom_model(data):
    """
    Function to load a custom model.

    Parameters:
    data (dict): dictionary with the model name, model custom name, and model custom parameters

    Returns:
    model_out (nn.Module): loaded custom model
    """
    model_name = data['model_name']
    model_custom = data['model_custom']
    model_module = importlib.import_module(model_custom)
    if model_name=='Unet': ## Dynamic Unet with attention - not implemented in the new update
        if data['unet_parameters']['attention']:
            model_name = 'Unet_attention'
        model_out = getattr(model_module, model_name)(**data['unet_parameters'])
    else:
        model_out = getattr(model_module, model_name)(**data['model_custom_params'])
    return model_out
    

class ModelLoader(nn.Module):
    def __init__(self, data : dict,logger=None):
        """
        Initialize the ModelLoader class.

        Parameters
        ----------
        data : dict
            A dictionary containing the following information:
                - model_name: str
                    The name of the model to use.
                - model_path: str
                    The path to the model weights file.
                - model_pretrained: bool
                    Whether to use a pre-trained model.
                - load_model: bool
                    Whether to load the model weights from the specified path.
                - model_num_classes: int
                    The number of classes to predict.
                - model_type: str
                    The type of model to use (e.g. classification, detection, segmentation).
                - use_unet: bool
                    Whether to use a U-Net model for segmentation tasks.
        logger : Logger or None
            The logger to use for printing messages.
        """
        super().__init__()
        self._data= data 
        self._model_name=self._data['model_name']
        self._model_path=self._data['model_path']
        self._model_pretrained=self._data['model_pretrained']
        self._load_model = self._data['load_model']
        self._model_num_classes = self._data['model_num_classes']
        self._model_type=self._data['model_type']
        # self._model_unet= self._data['use_unet']

    def model_type(self):
        """
        Function to get default model resnet, vgg, densenet, googlenet, inception, mobilenet, mnasnet, shufflenet_v2, squeezenet, or object_detection
        """

        if self._model_type=='detection':
            model_out = getattr(torchvision.models.detection,self._model_name)(pretrained=self._model_pretrained)
            return model_out


        if self._model_type=='classification':
            model_out = getattr(torchvision.models,self._model_name)(pretrained=self._model_pretrained)
            if hasattr(model_out,'fc'):
                num_ftrs = model_out.fc.in_features
                model_out.fc = nn.Linear(num_ftrs, self._model_num_classes)            
            if hasattr(model_out,'classifier'):
                for module in list(model_out.modules()):
                    if isinstance(module, nn.Linear):
                        first_layer = module
                        num_ftrs = first_layer.in_features
                        model_out.classifier = nn.Linear(num_ftrs, self._model_num_classes)
                        break
            return model_out
        
    def get_model(self):
        """
        FUnction to get the model
        """
        # Check if the model is available in torchvision models

        # if self._model_unet:
        #     raise NotImplementedError("Unet not implemented in the new update.")

        if self._load_model:
            self.model = torch.load(self._data['load_model'])
            return self.model

        try:
            # Try to load the model from the specified path
            if hasattr(models, self._model_name) or hasattr(torchvision.models.detection, self._model_name):
                self.model = self.model_type() 
                if logger:
                    logger.info(f"Model {self._model_name} loaded from torchvision.models.") 
                return self.model
        except FileNotFoundError:
            raise ValueError(f"Model {self._model_name} not found in torchvision.models.")
    
    def forward(self,x):
        return self.model(x)
    
class LayerExtractor:
    def __init__(self, model, dataloader, target_layers : list ,device='cpu'):
        """
        Initialize the LayerExtractor class.

        Args:
            model (nn.Module): Model which we want to extract the features from.
            dataloader (DataLoader): Dataloader which will be used to extract the features.
            target_layers (list): List of target layers from which the features are to be extracted.
            device (str, optional): Device on which the model is to be run. Defaults to 'cpu'.
        """
        self.model = model
        self.dataloader = dataloader
        self.target_layers = target_layers
        self.device = device
        self._features = {}
        
        model_layers = dict(self.model.named_modules())
        for layer in self.target_layers:
            if layer not in model_layers:
                raise ValueError(f"Layer '{layer}' not found in the model")

        for name, module in self.model.named_modules():
            if name in self.target_layers:
                module.register_forward_hook(self._hook_fn(name))

    def _hook_fn(self, layer_name):
        """
        Creates a forward hook function for a specified layer.

        This hook function captures the output of the layer during the forward pass
        and stores it in a dictionary with the layer name as the key. The output
        is reshaped and converted to a NumPy array before storing.

        Args:
            layer_name (str): The name of the layer to attach the forward hook to.

        Returns:
            function: A hook function that can be registered to a model layer.
        """

        def hook(module, input, output):
            N, C, H, W = output.shape
            output = output.reshape(N, C, -1)
            self._features[layer_name]=output.data.cpu().numpy()
        return hook

    def forward(self, x):
        x = x.to(self.device)
        _ = self.model(x)  # Run the forward pass
        return self._features  # Return the hooked layer's features
    
    def generate_embeddings(self):
        self.model.to(self.device)
        self.model.eval()

        final_features = {i_layer: [] for i_layer in self.target_layers}

        for i, i_dat in tqdm(enumerate(self.dataloader)):
            # Reset _features for each forward pass
            self._features = {i_layer: None for i_layer in self.target_layers}
            
            # Forward pass
            _ = self.model(i_dat['image'])
            
            # Append features for each layer
            for layer_name in self.target_layers:
                final_features[layer_name].append(self._features[layer_name])

        return final_features
        