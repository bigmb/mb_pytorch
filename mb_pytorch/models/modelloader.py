from mb_utils.src.logging import logger
from torch import nn
import torch
from torch.nn import functional as F
import torchvision
import torchvision.models as models
import os
import importlib

__all__ = ['ModelLoader','ModelExtractor']


def get_custome_model(data):
    """
    Function to get custom model from the models folder
    """
    model_name = data['model_name']
    model_custom = data['model_custom']
    model_module = importlib.import_module(model_custom)
    if model_name=='Unet':
        if data['unet_parameters']['attention']:
            model_name = 'Unet_attention'
    model_class = getattr(model_module, model_name)   
    model_out = model_class(**data['unet_parameters'])
    return model_out
    

class ModelLoader(nn.Module):
    def __init__(self, data : dict,logger=None):
        super().__init__()
        self._data= data 
        self._use_torchvision_models=self._data['use_torchvision_models']
        self._model_name=self._data['model_name']
        self._model_version=self._data['model_version']
        self._model_path=self._data['model_path']
        self._model_pretrained=self._data['model_pretrained']
        self._load_model = self._data['load_model']
        self._model_num_classes = self._data['model_num_classes']
        self._model_type=self._data['model_type']
        if self._model_version==None:
            self._model_version=''
        self._model_final_name = self._model_name + self._model_version

    def model_type(self):
        """
        Function to get default model resnet, vgg, densenet, googlenet, inception, mobilenet, mnasnet, shufflenet_v2, squeezenet
        """
        model_out = getattr(torchvision.models,self._model_final_name)(pretrained=self._model_pretrained)
        if self._model_type=='classification':
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
            #model_out.softmax = nn.Softmax(dim=1)
        
            
        #     for module in reversed(list(model_out.modules())):
        #         if isinstance(module, nn.Linear):
        #             last_layer = module
        #             num_ftrs = last_layer.in_features
        #             model_out.classifier = nn.Linear(num_ftrs, self._model_num_classes)
        #             break
        return model_out

    def model_params(self):
        """
        Function to pass the model params to custom model
        """        
        #check if model is available in the models list
        model_out = get_custome_model(self._data)
        return model_out
        

    def get_model(self):
        """
        FUnction to get the model
        """
        # Check if the model is available in torchvision models

        if self._load_model:
            self.model = torch.load(self._data['load_model'])
            return self.model

        if self._use_torchvision_models:
            try:
                # Try to load the model from the specified path
                if hasattr(models, self._model_final_name):
                    #model_class = getattr(models, self._model_name)
                    #if self._model_name in ['resnet', 'vgg', 'densenet', 'googlenet', 'inception', 'mobilenet', 'mnasnet', 'shufflenet_v2', 'squeezenet']:
                        # These models have pretrained weights available
                    self.model = self.model_type() 
                    if logger:
                        logger.info(f"Model {self._model_name} loaded from torchvision.models.") 
            except FileNotFoundError:
                raise ValueError(f"Model {self._model_name} not found in torchvision.models.")

        else:
            self.model = self.model_params()
        return self.model
    
    def forward(self,x):
        return self.model(x)
    

class ModelExtractor(nn.Module):
    def __init__(self, model):
        super(ModelExtractor, self).__init__()
        self.model = model
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
                
    def __call__(self, x):
        return self.feature_extractor(x)[:, :, 0, 0]