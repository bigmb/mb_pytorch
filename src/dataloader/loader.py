#dataloader for pytorch1.0

import torchdata
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from src.utils.yaml_reader import YamlReader

__all__ = ['data_fetcher','DataLoader']

class data_fetcher:
    """
    dataloader for pytorch1.0
    """
    def __init__(self,yaml,logger=None) -> None:
        self.yaml = yaml
        self.logger = logger
        self._yaml_data = None
        self.data_dict = {}
        self.transforms_final=[]

    @property
    def read_yaml(self):
        """
        read yaml file
        """
        self._yaml_data = YamlReader(self.yaml).data(self.logger)
        return self._yaml_data

    @property
    def load_data_params(self):
        """
        get dataloader data from yaml file
        """
        data = YamlReader(self.yaml).data(self.logger)
        self.data_dict['data'] = data['data']
        self.data_dict['transforms_list'] = data['transformation']
        return self.data_dict
    
    @property
    def get_transforms(self):
        """
        get transforms from yaml file
        """
        transforms_list = self.load_data_params['transforms_list']

        # for t_list in transforms_list:
        #     if t_list in dir(transforms):
        #         self.transforms_final.append(transforms.t_list)

        if len(self.transforms_final) != 0:
            self.transforms_final = []

        #for t_list in transforms_list:
        if transforms_list['to_tensor']['val']:
            self.transforms_final.append(transforms.ToTensor())
        if transforms_list['normalize']['val']:
            self.transforms_final.append(transforms.Normalize(transforms_list['normalize']['args']['mean'],transforms_list['normalize']['args']['std']))
        if transforms_list['resize']['val']:
            self.transforms_final.append(transforms.Resize(transforms_list['resize']['args']['size']))
        if transforms_list['random_crop']['val']:
            self.transforms_final.append(transforms.RandomCrop(transforms_list['random_crop']['args']['size']))
        if transforms_list['random_horizontal_flip']['val']:
            self.transforms_final.append(transforms.RandomHorizontalFlip(transforms_list['random_horizontal_flip']['args']['p']))
        if transforms_list['random_vertical_flip']['val']:
            self.transforms_final.append(transforms.RandomVerticalFlip(transforms_list['random_vertical_flip']['args']['p']))
        if transforms_list['random_rotation']['val']:
            self.transforms_final.append(transforms.RandomRotation(transforms_list['random_rotation']['args']['degrees']))
        if transforms_list['random_color_jitter']['val']:
            self.transforms_final.append(transforms.ColorJitter(transforms_list['random_color_jitter']['args']['brightness'],transforms_list['random_color_jitter']['args']['contrast'],transforms_list['random_color_jitter']['args']['saturation'],transforms_list['random_color_jitter']['args']['hue']))
        if transforms_list['random_grayscale']['val']:
            self.transforms_final.append(transforms.RandomGrayscale(transforms_list['random_grayscale']['args']['p']))
        if self.logger:
            self.logger.info("transforms: {}".format(self.transforms_final))
        return self.transforms_final
   

class DataLoader(data_fetcher):
    def __init__(self,yaml,logger=None) -> None:
        super().__init__(yaml, logger=logger)
        self.yaml = yaml
        self.logger = logger
        self._yaml_data = None
        self.data_dict = {}
        self.transforms_final=[]
        self.trainset = None
        self.trainloader = None
    
    def data_load(self,data_file = 'CIFAR10',embeddings=False,logger=None):
        """
        return all data loaders
        """
        if not self.trainset:
            self.trainset = self.data_train(data_file,transform=self.get_transforms,logger=self.logger,collate_fn=None)
        if not self.trainloader:
            self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.load_data()['data_train']['batch_size'], shuffle=self.load_data()['data_train']['shuffle'], num_workers=self.load_data()['data_train']['num_workers'], collate_fn=None)
        if not self.testloader:
            self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.load_data()['data_test']['batch_size'], shuffle=self.load_data()['data_test']['shuffle'], num_workers=self.load_data()['data_test']['num_workers'],collate_fn=None)
        if not self.valloader:
            self.valloader = torch.utils.data.DataLoader(self.valset, batch_size=self.load_data()['data_val']['batch_size'], shuffle=self.load_data()['data_val']['shuffle'], num_workers=self.load_data()['data_val']['num_workers'],collate_fn=None)

        return self.trainloader,self.testloader,self.valloader


    def data_train(self,data_file,transform,logger=None):
        """
        get train data from yaml file
        """
        #datasets.CIFAR10(data_path, train=True, download=False,transform=transform)