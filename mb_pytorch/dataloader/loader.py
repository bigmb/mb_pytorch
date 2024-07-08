#dataloader for pytorch1.0

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from mb_pytorch.utils.yaml_reader import YamlReader
import os
import numpy as np
import pandas as pd
from mb_pandas.src.dfload import load_any_df
from mb_utils.src.verify_image import verify_image
from mb.pandas import check_drop_duplicates,remove_unnamed
from datetime import datetime
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
        self.data_dict['transformation'] = data['transformation']
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


class JointTransforms:
    def __init__(self,transform_yaml,logger=None):
        """
        get transforms from yaml file
        """

        self.transform_data = transform_yaml
        self.logger = logger

        if self.transform_data['transform']==False:  # noqa: E712
            return None

    def __call__(self,img,mask=None,bbox=None):
        if self.transform_data['to_tensor']['val']:
            img = transforms.ToTensor()(img)
            if mask is not None:
                mask = transforms.ToTensor()(mask)
            if bbox is not None:
                bbox = torch.tensor([bbox[0],bbox[1],bbox[2],bbox[3]],dtype=torch.int32)

        if self.transform_data['normalize']['val']:
            img = transforms.Normalize(self.transform_data['normalize']['args']['mean'],self.transform_data['normalize']['args']['std'])(img)

        if self.transform_data['resize']['val']:
            ori_size = img.size()
            img = transforms.Resize(self.transform_data['resize']['args']['size'])(img)
            if mask is not None:
                mask = transforms.Resize(self.transform_data['resize']['args']['size'])(mask)
            if bbox is not None:
                bbox = self.resize_boxes(ori_size,img.size(),bbox)

        if self.transform_data['random_crop']['val']:
            img = transforms.RandomCrop(self.transform_data['random_crop']['args']['size'])(img)
            if mask is not None:
                mask = transforms.RandomCrop(self.transform_data['random_crop']['args']['size'])(mask)
            if bbox is not None:
                bbox = self.crop_boxes(bbox, *self.transform_data['random_crop']['args']['size'])

        if self.transform_data['random_horizontal_flip']['val']:
            img = transforms.RandomHorizontalFlip(self.transform_data['random_horizontal_flip']['args']['p'])(img)
            if mask is not None:
                mask = transforms.RandomHorizontalFlip(self.transform_data['random_horizontal_flip']['args']['p'])(mask)
            if bbox is not None:
                bbox = self.hflip_boxes(bbox, img.size()[1])

        if self.transform_data['random_vertical_flip']['val']:
            img = transforms.RandomVerticalFlip(self.transform_data['random_vertical_flip']['args']['p'])(img)
            if mask is not None:
                mask = transforms.RandomVerticalFlip(self.transform_data['random_vertical_flip']['args']['p'])(mask)
            if bbox is not None:
                bbox = self.vflip_boxes(bbox, img.size()[2])

        if self.transform_data['random_rotation']['val']:
            img = transforms.RandomRotation(self.transform_data['random_rotation']['args']['degrees'])(img)
            if mask is not None:
                mask = transforms.RandomRotation(self.transform_data['random_rotation']['args']['degrees'])(mask)
            if bbox is not None:
                bbox = self.rotate_boxes(bbox, self.transform_data['random_rotation']['args']['degrees'], img.size[1], img.size[0])

        if self.transform_data['random_color_jitter']['val']:
            img = transforms.ColorJitter(brightness=self.transform_data['random_color_jitter']['args']['brightness'],contrast=self.transform_data['random_color_jitter']['args']['contrast'],saturation=self.transform_data['random_color_jitter']['args']['saturation'],hue=self.transform_data['random_color_jitter']['args']['hue'])(img)
        
        if self.transform_data['random_grayscale']['val']:
            img = transforms.RandomGrayscale(self.transform_data['random_grayscale']['args']['p'])(img)
        
        if mask is not None:
            return img,mask
        elif bbox is not None:
            return img,bbox
        else:
            return img
        
    def resize_boxes(self,original_size, new_size, bbox):
        """
        Resize bounding boxes according to the new image size
        Input:
        original_size: torch.tensor, (channel, original_height, original_width)
        new_size: tuple, (channel, new_height,new_width)
        bbox: list, [x1, y1, x2, y2]
        """
        orig_x1, orig_y1, orig_x2, orig_y2 = bbox
        _, orig_height,orig_width = original_size
        _, new_height, new_width = new_size
    
        x_scale = new_width / orig_width
        y_scale = new_height / orig_height
    
        new_x1 = int(orig_x1 * x_scale)
        new_y1 = int(orig_y1 * y_scale)
        new_x2 = int(orig_x2 * x_scale)
        new_y2 = int(orig_y2 * y_scale)
    
        return torch.tensor([new_x1, new_y1, new_x2, new_y2],dtype=torch.int32)

    def crop_boxes(self, boxes, top, left, height, width):
        boxes[:, [0, 2]] = boxes[:, [0, 2]] - left
        boxes[:, [1, 3]] = boxes[:, [1, 3]] - top

        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=height)
        return boxes

    def hflip_boxes(self, boxes, image_width):
        boxes[:, [0, 2]] = image_width - boxes[:, [2, 0]]
        return boxes
   
    def vflip_boxes(self, boxes, image_height):
        boxes[:, [1, 3]] = image_height - boxes[:, [3, 1]]
        return boxes
    
    def rotate_boxes(self, boxes, angle, image_height, image_width):
        # Convert the angle to radians
        angle = -angle * np.pi / 180.0
        boxes = self.rotate_polygon(boxes, angle, image_height, image_width)
        return boxes

    def rotate_polygon(self, polygon, angle, image_height, image_width):
        # Get the center of the polygon
        center = polygon.mean(axis=0)

        # Shift the polygon so that the center of the polygon is at the origin
        shifted_polygon = polygon - center

        # Rotate the polygon
        rotated_polygon = self.rotate_point(shifted_polygon, angle)

        # Shift the polygon back
        rotated_polygon += center

        return rotated_polygon
    
    def rotate_point(self, point, angle):
        # Get the rotation matrix
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return np.dot(point, rotation_matrix)


class customdl(torch.utils.data.Dataset):
    def __init__(self,data,model_type,transform=None,train_file=True,logger=None):
        self.transform=transform
        self.logger=logger
        self.folder_name=os.path.dirname(data['root'])
        self.data_type = model_type
        self.csv_data = load_any_df(data['root'],logger=self.logger)

        if self.logger:
            self.logger.info("Data file: {} loaded with mb_pandas.".format(data))
            self.logger.info("Data columns: {}".format(self.csv_data.columns))
            self.logger.info("Data will be split into train and validation according to train_file input : {}".format(train_file))
            self.logger.info("If unnamed columns are present, they will be removed.")
            self.logger.info("If duplicate rows are present, they will be removed.")
        assert 'image_path' in self.csv_data.columns, "image_path column not found in data"
        assert 'image_type' in self.csv_data.columns, "image_type column not found in data"

        #checking paths
        path_check_res= [os.path.exists(self.csv_data.image_path[i]) for i in range(len(self.csv_data))]
        self.csv_data['img_path_check'] = path_check_res ## check why test loader doesnt get this column
        self.csv_data = self.csv_data[self.csv_data['img_path_check'] == True]  # noqa: E712
        self.csv_data = self.csv_data.reset_index(drop=True)
        if logger:
            self.logger.info("Length of data after removing invalid paths: {}".format(len(self.csv_data)))

        if data['verify_image']:
            if self.logger:
                self.logger.info("Verifying images")
            verify_image_res = [verify_image(self.csv_data['image_path'].iloc[i],logger=self.logger) for i in range(len(self.csv_data))]  
            self.csv_data['img_verify'] = verify_image_res
            self.csv_data = self.csv_data[self.csv_data['img_verify'] == True]  # noqa: E712
            self.csv_data = self.csv_data.reset_index()
        else:   
            if self.logger:
                self.logger.info("Skipping image verification")

        if train_file: ## used this to differentiate between train and validation data in the data file
            try:
                if 'training' in pd.unique(self.csv_data['image_type']):
                    self.csv_data = self.csv_data[self.csv_data['image_type'] == 'training']
                if 'train' in pd.unique(self.csv_data['image_type']):   
                    self.csv_data = self.csv_data[self.csv_data['image_type'] == 'train']
            except Exception as e:
                if self.logger:
                    self.logger.info(e)
                return "image_type column (train/training) not found in data"
        else:
            try:
                if 'validation' in pd.unique(self.csv_data['image_type']):
                    self.csv_data = self.csv_data[self.csv_data['image_type'] == 'validation']
                if 'val'  in pd.unique(self.csv_data['image_type']):
                    self.csv_data = self.csv_data[self.csv_data['image_type'] == 'val']
                if 'test' in pd.unique(self.csv_data['image_type']):
                    self.csv_data = self.csv_data[self.csv_data['image_type'] == 'test']
                if 'testing' in pd.unique(self.csv_data['image_type']):
                    self.csv_data = self.csv_data[self.csv_data['image_type'] == 'testing']
            except Exception as e:
                if self.logger:
                    self.logger.info(e)
                return "image_type column (val/validation/test/testing) not found in data"

        self.csv_data = check_drop_duplicates(self.csv_data,columns=['image_path'],drop=True,logger=self.logger)
        self.csv_data = remove_unnamed(self.csv_data,logger=self.logger)
        if logger:
            self.logger.info("Length of data after removing duplicates and unnamed columns: {}".format(len(self.csv_data)))
        
        if self.data_type == 'classification':
            assert 'label' in self.csv_data.columns, "label column not found in data"
            self.label = self.csv_data['label']
    
        if self.data_type == 'segmentation':
            assert 'mask_path' in self.csv_data.columns, "mask_path column not found in data"
            self.masks = self.csv_data['mask_path']

        if self.data_type == 'detection':
            assert 'label' in self.csv_data.columns, "label column not found in data"
            assert 'bbox' in self.csv_data.columns, "bbox column not found in data"
            self.label = self.csv_data['label']
            self.bbox = self.csv_data['bbox']

        ## save wrangled file
        if train_file:
            try:
                if os.path.exists(self.folder_name):
                    self.csv_data.to_csv(os.path.join(self.folder_name,'train_wrangled_file.csv'),index=False)
            except Exception as e:
                if self.logger:
                    self.logger.info(e)
                    self.logger.info("Could not save wrangled file. Please check the folder name.")
        else:
            try:
                if os.path.exists(self.folder_name):
                    self.csv_data.to_csv(os.path.join(self.folder_name,'val_wrangled_file.csv'),index=False)
            except Exception as e:
                if self.logger:
                    self.logger.info(e)
                    self.logger.info("Could not save wrangled file. Please check the folder name.")

    def __len__(self):
        return len(self.csv_data)
    
    def __repr__(self) -> str:
        return "self.data: {},self.transform: {},self.label: {}".format(self.csv_data,self.transform,self.label)

    def __getitem__(self,idx):
        
        img = self.csv_data['image_path'].iloc[idx]
        #img = Image.open(img)
        img = cv2.imread(img)

        if self.data_type == 'classification':
            if self.transform:
                img = self.transform(img)
            label = {}
            label['label'] = self.label.iloc[idx]   
            return img,label
        
        if self.data_type == 'segmentation':
            if self.transform:
                mask = cv2.imread(self.masks.iloc[idx],cv2.IMREAD_GRAYSCALE) ## considering mask is just binary class
                img,mask = self.transform(img,mask=mask)
            mask_dict={}
            mask_dict['mask'] = mask
            mask_dict['label'] = self.label.iloc[idx]
            
            return img,mask_dict
        
        if self.data_type == 'detection':
            bbox = eval(self.bbox.iloc[idx])
            if self.transform:
                img,bbox = self.transform(img,bbox=bbox)
            else:
                img = transforms.ToTensor()(img)
                bbox = torch.tensor([[bbox[0],bbox[1],bbox[2],bbox[3]]],dtype=torch.int32)
            bbox_dict={}
            bbox_dict['bbox'] = bbox
            bbox_dict['label'] = [self.label.iloc[idx]]

            return img,bbox_dict

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
        self.trainloader = None
        self.testloader = None
        self.model_type = self.data_dict['model']['model_type']
        self.transformations = self.data_dict['transformation']
        self.data_params_file = self.data_dict['data']['file']

    
    def data_load(self):
        """
        return all data loaders
        """

        self.trainset = self.data_train(self.data_params_file,self.model_type, 
                                        transform=JointTransforms(self.transformations),train_file=True,logger=self.logger)
        self.testset = self.data_train(self.data_params_file,self.model_type,
                                        transform=JointTransforms(self.transformations),train_file=False,logger=self.logger)

        if self.model_type == 'classification':
            def collate_fn(batch):
                return {'image': torch.stack([b[0] for b in batch]),
                         'label': torch.tensor([b[1]['label'] for b in batch])}
        elif self.model_type == 'segmentation':
            def collate_fn(batch):
                return {'image': torch.stack([b[0] for b in batch]),
                         'mask': torch.stack([b[1]['mask'] for b in batch]), 
                        'label': torch.tensor([b[1]['label'] for b in batch])}
        elif self.model_type == 'detection':
            def collate_fn(batch):
                return {'image': torch.stack([b[0] for b in batch]),
                         'bbox': torch.stack([b[1]['bbox'] for b in batch]), 
                        'label': torch.tensor([b[1]['label'] for b in batch])} ## label is a list of labels or should be hot encoded?


        self.trainloader = torch.utils.data.DataLoader(self.trainset, 
                                                       batch_size=self.data_dict['train_params']['batch_size'], 
                                                       shuffle=self.data_dict['train_params']['shuffle'], 
                                                       num_workers=self.data_dict['train_params']['num_workers'],
                                                       worker_init_fn = lambda id: np.array(self.data_dict['train_params']['seed']),
                                                       collate_fn=collate_fn)
        self.testloader = torch.utils.data.DataLoader(self.testset, 
                                                      batch_size=self.data_dict['test_params']['batch_size'], 
                                                      shuffle=self.data_dict['test_params']['shuffle'], 
                                                      num_workers=self.data_dict['test_params']['num_workers'],
                                                      worker_init_fn = lambda id: np.array(self.data_dict['test_params']['seed']),
                                                      collate_fn=collate_fn)
        return self.trainloader,self.testloader,self.trainset,self.testset

    
    def data_train(self,data,model_type,transform=None,train_file=True,**kwargs):
        """
        get train data from yaml file
        """
        data_t = customdl(data,model_type,transform=transform,train_file=train_file,**kwargs)
        return data_t