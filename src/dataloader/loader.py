#dataloader for pytorch1.0

import torchdata
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from src.utils.yaml_reader import YamlReader
import os
from mb_pandas.src.dfload import load_any_df
from mb_utils.src.verify_image import verify_image
from mb_pandas.src.transform import *
from datetime import datetime

today = datetime.now()

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
   
class customdl(torch.utils.data.Dataset):
    def __init__(self,data,folder_name=None,transform=None,logger=None):
        self.data=load_any_df(data)
        self.transform=transform
        self.logger=logger

        if self.logger:
            self.logger.info("Data file: {} loaded with mb_pandas.".format(data))
            self.logger.info("Data columns: {}".format(self.data.columns))
            self.logger.info("If unnamed columns are present, they will be removed.")
            self.logger.info("If duplicate rows are present, they will be removed.")
        assert 'image_path' in self.data.columns, "image_path column not found in data"
        assert 'label' in self.data.columns, "label column not found in data"
        assert 'image_type' in self.data.columns, "image_type column not found in data"

        self.data = remove_unnamed(self.data,logger=self.logger)
        self.data = check_drop_duplicates(self.data,columns=['image_path'],drop=True,logger=self.logger)

        if folder_name:
            self.folder_name=folder_name
        # else:
        #     date_now = today.strftime("%d_%m_%Y_%H_%M")
        #     self.folder_name='data_'+date_now
        # os.mkdir('./data'+str(self.folder_name))
        img_path = [os.path.join(str(self.folder_name),self.data['image_path'].iloc[i]) for i in range(len(self.data))]
        self.data['image_path'] = img_path
        if self.logger:
            self.logger.info("Verifying images")
        self.data = verify_image(self.data,logger=self.logger)

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        
                                                                
        return {self.label[idx]:self.emb[idx]}

class DataLoader(data_fetcher):
    """
    Basic dataloader for pytorch1.0
    """
    def __init__(self,yaml,logger=None) -> None:
        super().__init__(yaml, logger=logger)
        self.yaml = yaml
        self.logger = logger
        self._yaml_data = None
        self.data_dict = {}
        self.transforms_final=[]
        self.trainloader = None
        self.testloader = None
    
    def data_load(self,data_file = 'CIFAR10',embeddings=False,logger=None):
        """
        return all data loaders
        """

        if data_file in dir(torchvision.datasets):
            if self.logger:
                self.logger.info("Data file: {} loading from torchvision.datasets.".format(data_file))
            if data_file in os.listdir('../../data/'):
                download_flag = False
            else:
                download_flag = True
            self.trainset = getattr(torchvision.datasets,data_file)(root='../../data/', train=True, download=download_flag,transform=self.get_transforms)
            self.testset = getattr(torchvision.datasets,data_file)(root='../../data/', train=False, download=download_flag,transform=self.get_transforms)
        else:
            self.trainset = self.data_train(data_file,transform=self.get_transforms,logger=self.logger)
            
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.load_data()['data_train']['batch_size'], shuffle=self.load_data()['data_train']['shuffle'], num_workers=self.load_data()['data_train']['num_workers'])
        return self.dataloader


    def data_train(self,data_file,transform,logger=None):
        """
        get train data from yaml file
        """
