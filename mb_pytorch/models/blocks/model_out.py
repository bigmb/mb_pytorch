from torch import nn
import torch
from torch.nn import functional as F
import torchvision
import torchvision.models as models
from mb_pytorch.models.blocks.attention_block import Attention
from mb_pytorch.models.blocks.conv_block import Conv

__all__ = ['ModelOut']

class ModelOut(nn.Modules):
    def __init__(self, yaml_data,logger=None):
        self._data=yaml_data
        self._use_torchvision_models=self._data['use_torchvision_models']
        self._model_name=self._data['model_name']
        self._model_version=self._data['model_version']
        self._model_path=self._data['model_path']

    def model_type(model_name,model_version,pretrained=False):
        """
        Function to get default model resnet, vgg, densenet, googlenet, inception, mobilenet, mnasnet, shufflenet_v2, squeezenet
        """
        model_final = model_name+model_version
        model_out = getattr(models.model_name,model_final)(pretrained=pretrained)
        return model_out

    def get_model(self, model_name, model_version, model_path=None):
        """
        Function to get the model architecture from torchvision.models or from a specified path
        Inputs:
            model_name (str): Name of the model architecture
            model_path (str): Path to the model architecture
        Returns:
            model (torch.nn.Module): Model architecture
        """
        # Check if the model is available in torchvision models

        if self._use_torchvision_models==False:
            try:
                # Try to load the model from the specified path
                self.model = model_name
            except FileNotFoundError:
                raise ValueError(f"Model {model_name} not found in torchvision.models and path {model_path} does not exist.")

        else:
            if hasattr(models, model_name):
                model_class = getattr(models, model_name)
                if model_name in ['resnet', 'vgg', 'densenet', 'googlenet', 'inception', 'mobilenet', 'mnasnet', 'shufflenet_v2', 'squeezenet']:
                    # These models have pretrained weights available
                    self.model = self.model_type(model_name,model_version,pretrained=True)    

    
    def forward(self,x):
        return self.model(x)