#dataloader for pytorch1.0

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from mb_pytorch.utils.yaml_reader import YamlReader
import os
import sys
import numpy as np
from mb_pandas.src.dfload import load_any_df
from mb_utils.src.verify_image import verify_image
from mb_pandas.src.transform import *
from datetime import datetime
from PIL import Image
import cv2

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
        self.all = None

    def __repr__(self) -> str:
        return "data_fetcher(yaml={},logger={})".format(self.yaml,self.logger)

    @staticmethod
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
        self.data_dict['train_params'] = data['train_params']
        self.data_dict['test_params'] = data['test_params']
        self.data_dict['transforms_list'] = data['transformation']
        self.data_dict['model'] = data['model']
        return self.data_dict
    
    @property
    def load_data_all(self):
        """
        get dataloader all data dict from yaml file
        """
        data = YamlReader(self.yaml).data(self.logger)
        self.all = data
        return self.all

    @property
    def get_transforms(self):
        """
        get transforms from yaml file
        """

        transforms_list = self.load_data_params['transforms_list']

        if transforms_list['transform']==False:
            return None

        # for t_list in transforms_list:
        #     if t_list in dir(transforms):
        #         self.transforms_final.append(transforms.t_list)

        if 'transforms' in dir(self.transforms_final):
            self.transforms_final = []

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
        
        self.transforms_final = transforms.Compose(self.transforms_final)
        return self.transforms_final
   
class customdl(torch.utils.data.Dataset):
    def __init__(self,data,transform=None,train_file=True,logger=None):
        self.transform=transform
        self.logger=logger
        self.folder_name=data['work_dir']
        self.data = load_any_df(data['file'],logger=self.logger)

        if self.logger:
            self.logger.info("Data file: {} loaded with mb_pandas.".format(data))
            self.logger.info("Data columns: {}".format(self.data.columns))
            self.logger.info("Data will be split into train and validation according to train_file input : {}".format(train_file))
            self.logger.info("If unnamed columns are present, they will be removed.")
            self.logger.info("If duplicate rows are present, they will be removed.")
        assert 'image_path' in self.data.columns, "image_path column not found in data"
        assert 'image_type' in self.data.columns, "image_type column not found in data"

        if train_file:
            self.data = self.data[self.data['image_type'] == 'training']
        else:
            self.data = self.data[self.data['image_type'] == 'validation']

        if 'label' in self.data.columns:
            self.label = self.data['label']
        else:
            self.label = None

        self.data = check_drop_duplicates(self.data,columns=['image_path'],drop=True,logger=self.logger)
        self.data = remove_unnamed(self.data,logger=self.logger)

        # else:
        #     date_now = today.strftime("%d_%m_%Y_%H_%M")
        #     self.folder_name='data_'+date_now
        # os.mkdir('./data'+str(self.folder_name))

        if data['use_img_dir']:
            img_path = [os.path.join(str(data['img_dir']),self.data['image_path'].iloc[i]) for i in range(len(self.data))]
        else:
            img_path = [self.data['image_path'].iloc[i] for i in range(len(self.data))]
        self.data['image_path_new'] = img_path
        if self.logger:
            self.logger.info("Verifying paths")
            self.logger.info("first path : {}".format(img_path[0]))

        path_check_res= [os.path.exists(img_path[i]) for i in range(len(img_path))]
        self.data['img_path_check'] = path_check_res
        self.data = self.data[self.data['img_path_check'] == True]
        self.data = self.data.reset_index(drop=True)
        if logger:
            self.logger.info("self.data: {}".format(self.data))

        if data['thresholding_pd']>0:
            if len(self.data) <= data['thresholding_pd']:
                self.logger.info("Length of data after removing invalid paths: {}".format(len(self.data)))
                self.logger.info("Less than thresholding_pd data points. Please check the data file.")
                self.logger.info("Exiting")
                sys.exit('Less than thresholding_pd data points. Please check the data file.')

        if self.logger:
            self.logger.info("Length of data after removing invalid paths: {}".format(len(self.data)))
            self.logger.info("Verifying images")
        verify_image_res = [verify_image(self.data['image_path_new'].iloc[i],logger=self.logger) for i in range(len(self.data))]  
        self.data['img_verify'] = verify_image_res
        self.data = self.data[self.data['img_verify'] == True]
        self.data = self.data.reset_index()

        if data['thresholding_pd']>0:
            if len(self.data) <= data['thresholding_pd']:
                self.logger.info("Length of data after removing invalid images: {}".format(len(self.data)))
                self.logger.info("Less than thresholding_pd data points. Please check the data file.")
                self.logger.info("Exiting")
                sys.exit('Less than thresholding_pd data points. Please check the data file.')
        
        if os.path.exists(self.folder_name):
            self.data.to_csv(os.path.join(self.folder_name,'wrangled_file.csv'),index=False)

    def __len__(self):
        return len(self.data)
    
    def __repr__(self) -> str:
        return "self.data: {},self.transform: {},self.label: {}".format(self.data,self.transform,self.label)

    def __getitem__(self,idx):
        
        img = self.data['image_path_new'].iloc[idx]
        #img = Image.open(img)
        img = cv2.imread(img)
        if self.label:
            label = self.label[idx]
        
        if self.transform:
            img = self.transform(img)

        out_dict = {'image':img}
        if self.label:
            out_dict['label'] = label                                                
        return out_dict

class DataLoader(data_fetcher):
    """
    Basic dataloader for pytorch1.0
    """
    def __init__(self,yaml,logger=None) -> None:
        super().__init__(yaml, logger=logger)
        self.yaml = yaml
        self.logger = logger
        self._yaml_data = None
        self.data_dict = self.load_data_params
        self.transforms_final=[]
        self.trainloader = None
        self.testloader = None
        self.folder_name = self.data_dict['data']['work_dir']
        self.data_file= self.data_dict['data']['from_datasets']

        if os.path.exists(self.folder_name):
            if self.logger:
                self.logger.info("Data folder already exists. Using existing data folder :  {}".format(self.folder_name))
        else:
            os.mkdir(self.folder_name)
            if self.logger:
                self.logger.info("Data folder created : {}".format(self.folder_name))
    
    def data_load(self):
        """
        return all data loaders
        """

        if self.data_dict['data']['from_file']==False:
            if self.data_file in dir(torchvision.datasets):
                if self.logger:
                    self.logger.info("Data file: {} loading from torchvision.datasets.".format(self.data_file))
                if self.data_file in os.listdir(self.folder_name):
                    download_flag = False
                else:
                    download_flag = True
                if self.data_file == 'CIFAR10' or self.data_file == 'CIFAR100':
                    self.trainset = getattr(torchvision.datasets,self.data_file)(root=self.folder_name, train=True, download=download_flag,transform=self.get_transforms)
                    self.testset = getattr(torchvision.datasets,self.data_file)(root=self.folder_name, train=False, download=download_flag,transform=self.get_transforms)
                else:
                    self.trainset = getattr(torchvision.datasets,self.data_file)(root=self.folder_name, split='train', download=download_flag,transform=self.get_transforms)
                    self.testset = getattr(torchvision.datasets,self.data_file)(root=self.folder_name, split='val', download=download_flag,transform=self.get_transforms)
                if self.data_dict['data']['thresholding_pd']>0:
                    subset_indices = range(self.data_dict['data']['thresholding_pd'])
                    self.trainset = torch.utils.data.Subset(self.trainset, subset_indices)
                    self.testset = torch.utils.data.Subset(self.testset, subset_indices)
            else:
                if self.logger:
                    self.logger.info("Data file: {} could not be loaded from torchvision.datasets.".format(self.data_file))
                    self.logger.info("Exiting")
                sys.exit("Data file: {} could not be loaded from torchvision.datasets.".format(self.data_file))
        else:
            self.trainset = self.data_train(self.data_dict['data'],transform=self.get_transforms,train_file=True,logger=self.logger)
            self.testset = self.data_train(self.data_dict['data'],transform=self.get_transforms,train_file=False,logger=self.logger)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.data_dict['train_params']['batch_size'], shuffle=self.data_dict['train_params']['shuffle'], num_workers=self.data_dict['train_params']['num_workers'],worker_init_fn = lambda id: np.array(self.data_dict['train_params']['seed']))
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.data_dict['test_params']['batch_size'], shuffle=self.data_dict['test_params']['shuffle'], num_workers=self.data_dict['test_params']['num_workers'],worker_init_fn = lambda id: np.array(self.data_dict['test_params']['seed']))
        return self.trainloader,self.testloader,self.trainset,self.testset

    def data_train(self,data_file,transform,train_file,logger=None):
        """
        get train data from yaml file
        """
        data_t = customdl(data_file,transform=transform,train_file=train_file,logger=logger)
        return data_t