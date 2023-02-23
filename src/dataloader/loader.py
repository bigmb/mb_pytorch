#dataloader for pytorch2.0

import torchdata
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.yaml_reader import YamlReader


class DataLoader_pytorch:
    """
    dataloader for pytorch2.0
    """
    def __init__(self,yaml,logger=None) -> None:
        self.yaml = yaml
        self.logger = logger
        self._yaml_data = None
        self.data_dict = {}
        self.transforms_final=[]

    def load_data(self):
        """
        get dataloader data from yaml file
        """
        if not self._yaml_data:
            data = YamlReader(self.yaml).data(self.logger)[0]
            self.data_dict['data_train'] = data['data']['train']
            self.data_dict['data_val'] = data['data']['val']
            self.data_dict['data_test'] = data['data']['test']
            self.data_dict['transforms_list'] = data['transformation']
        return self.data_dict
    
    def get_transforms(self):
        """
        get transforms from yaml file
        """
        transforms_list = self.load_data()['transforms_list']
        for i in transforms_list:
            if transforms_list['to_tensor']['val']:
                self.transforms_final.append(transforms.ToTensor())
            if transforms_list['normalize']['val']:
                self.transforms_final.append(transforms.Normalize(transforms_list['normalize']['mean'],transforms_list['normalize']['std']))
            if transforms_list['resize']['val']:
                self.transforms_final.append(transforms.Resize(transforms_list['resize']['size']))
            if transforms_list['random_crop']['val']:
                self.transforms_final.append(transforms.RandomCrop(transforms_list['random_crop']['size']))
            if transforms_list['random_horizontal_flip']['val']:
                self.transforms_final.append(transforms.RandomHorizontalFlip(transforms_list['random_horizontal_flip']['p']))
            if transforms_list['random_vertical_flip']['val']:
                self.transforms_final.append(transforms.RandomVerticalFlip(transforms_list['random_vertical_flip']['p']))
            if transforms_list['random_rotation']['val']:
                self.transforms_final.append(transforms.RandomRotation(transforms_list['random_rotation']['degrees']))
            if transforms_list['random_color_jitter']['val']:
                self.transforms_final.append(transforms.ColorJitter(transforms_list['random_jitter']['brightness'],transforms_list['random_jitter']['contrast'],transforms_list['random_jitter']['saturation'],transforms_list['random_jitter']['hue']))
            if transforms_list['random_grayscale']['val']:
                self.transforms_final.append(transforms.RandomGrayscale(transforms_list['random_grayscale']['p']))
        if self.logger:
            self.logger.info("transforms: {}".format(self.transforms_final))
        return self.transforms_final
   
    def get_train_data(self):
        """
        get train data from yaml file
        """
        train_data = self.load_data()['data_train']
        train_data_final = torchvision.datasets.ImageFolder(root=train_data['root'],transform=transforms.Compose(self.get_transforms(self.logger)))
        if self.logger:
            self.logger.info("train data: {}".format(train_data_final))
        return train_data_final

        





