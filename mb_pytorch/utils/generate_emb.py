##Class for generating embeddings for the given data

import torch
from mb_pytorch.dataloader.loader import DataLoader
import cv2
import numpy as np
from tqdm import tqdm
from mb_pandas.src.dfload import load_any_df
from mb_utils.src.verify_image import verify_image
from mb_pandas.src.transform import *
import os
import torchvision
import torch.nn as nn
from torch.utils import hooks


__all__ = ['EmbeddingGenerator']


class customdl_emb(torch.utils.data.Dataset):
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

        self.data = check_drop_duplicates(self.data,columns=['image_path'],drop=True,logger=self.logger)
        self.data = remove_unnamed(self.data,logger=self.logger)

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
            self.logger.info("Length of data after removing invalid paths: {}".format(len(self.data)))
            self.logger.info("Verifying images")

        verify_image_res = [verify_image(self.data['image_path_new'].iloc[i],logger=self.logger) for i in range(len(self.data))]  
        self.data['img_verify'] = verify_image_res
        self.data = self.data[self.data['img_verify'] == True]
        self.data = self.data.reset_index()
        
        if os.path.exists(self.folder_name):
            self.data.to_csv(os.path.join(self.folder_name,'emb_wrangled_file.csv'),index=False)

    def __len__(self):
        return len(self.data)
    
    def __repr__(self) -> str:
        return "self.data: {},self.transform: {}".format(self.data,self.transform)

    def __getitem__(self,idx):
        
        img = self.data['image_path_new'].iloc[idx]
        #img = Image.open(img)
        img = cv2.imread(img)

        if self.transform:
            img = self.transform(img)

        out_dict = {'image':img}                                           
        return out_dict
    
class EmbeddingGenerator(DataLoader):
    def __init__(self, yaml, logger=None) -> None:
        super().__init__(yaml,logger=logger)
        
        self._data = self.load_data_all
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_file = self._data['data']['file']
        self.model = self._data['model']['model']
        self.use_pretrained = self._data['model']['use_pretrained']
        self.use_own_model = self._data['model']['use_own_model']
        if self.use_own_model:
            self.model = torch.load(self._data['model']['model_path'])

        self.ext_layer = self._data['model']['model_layer']
        self.transforms_final = self.get_transforms
        self._emb = None
        self.logger = logger

    def model_set(self,name=None):
        """
        Set the model for embedding generation. model already set in the yaml file
        """
        if name:
            k=eval("torchvision.models."+name)
        else:
            k=eval("torchvision.models."+self.model)
        self.model_val = k(pretrained=True)
        if self.logger:
            self.logger.info("Model set to {}".format(self._data['model']['model']))
        return self.model_val
    
    def generate_emb(self, data):
        """
        Generate embeddings for the given data
        Input:
            data: data for which embeddings are to be generated (numpy array)
        Output:
            emb: embeddings for the given data
        """
        model = self.model_set()
        model.to(self.device)
        model.eval()
        
        features_blobs = []
        def get_hook(module, input, output):
            N,C,H,W = output.shape
            output = output.reshape(N,C,-1)
            features_blobs.append(output.data.cpu().numpy())

        fea = model._modules.get(self.ext_layer).register_forward_hook(get_hook)
    
        if self.logger:
            self.logger.info("Embedding generation started")
            self.logger.info("length of data: {}".format(len(data)))

        for i,i_dat in tqdm(enumerate(data),total=len(data)):
            _ = model(i_dat['image'].to(self.device))

        self.emb= np.concatenate(features_blobs)
        self.emb = self.emb.reshape(self.emb.shape[0],self.emb.shape[1])

        fea.remove()
        del model
        if self.logger:
            self.logger.info("Embedding generation completed")
        return self.emb
    
    def file_save(self,logger=None):
        """
        Save the embeddings to a wrnagled file
        """
        work_dir = self._data['data']['work_dir']
        if os.path.exists(self.folder_name):
            df = load_any_df(os.path.join(self.folder_name,'emb_wrangled_file.csv'))
        df['embedding'] = self.emb.tolist()
        df.to_csv(os.path.join(self.folder_name,'emb_wrangled_file.csv'),index=False)
        if self.logger:
            self.logger.info("Embeddings saved to {}".format(os.path.join(self.folder_name,'emb_wrangled_file.csv')))


    def data_emb_loader(self):
        """
        get embedding data from yaml file
        """
        data = self.data_train_emb(self._data['data'],transform=self.get_transforms,train_file=False,logger=self.logger)
        loader = torch.utils.data.DataLoader(data, batch_size=self._data['train_params']['batch_size'],shuffle=self._data['train_params']['shuffle'],
                                              num_workers=self._data['train_params']['num_workers'])
        return loader

    def data_train_emb (self,data_file,transform,train_file,logger=None):
        """
        get train data from yaml file
        """
        data_t = customdl_emb(data_file,transform=transform,train_file=train_file,logger=logger)
        return data_t

        
