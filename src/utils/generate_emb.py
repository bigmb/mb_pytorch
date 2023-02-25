##Class for generating embeddings for the given data

import torch
from torch import optim, nn
from torchvision import models, transforms
from src.dataloader.loader import data_fetcher
import cv2
import numpy as np
from PIL import Image
import tqdm

__all__ = ['EmbeddingGenerator','FeatureExtractor']


class FeatureExtractor(nn.Module):
  def __init__(self, model):
    super(FeatureExtractor, self).__init__()

    self.features = list(model.features)
    self.features = nn.Sequential(*self.features)
    self.pooling = model.avgpool
    self.flatten = nn.Flatten()
    self.fc = model.classifier[0]
  
  def forward(self, x):
    out = self.features(x)
    out = self.pooling(out)
    out = self.flatten(out)
    out = self.fc(out) 
    return out 
  

class EmbeddingGenerator(data_fetcher):
    def __init__(self, yaml, logger=None) -> None:
        super().__init__(yaml,logger=logger)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_name = None
        self._data = self.load_data_params()
        self.transforms_final = self.get_transforms()
        self._emb = None
        self.logger = logger


    def model_set(self,name=None):
        """
        Set the model for embedding generation
        model already set in the yaml file
        """
        if name:
            k=eval("torchvision.models."+name)
        else:
            k=eval("torchvision.models."+self._data['emb']['model'])
        self.model = k(pretrained=True)
        if self.logger:
            self.logger.info("Model set to {}".format(self._data['emb']['model']))
        return self.model
    
    def generate_emb(self, data,transform=None):
        """
        Generate embeddings for the given data
        Input:
            data: data for which embeddings are to be generated (numpy array)
        Output:
            emb: embeddings for the given data
        """
        model = self.model_set()

        imgs = [cv2.imread(data[i]) for i in range(len(data))]
        if transform:
            #imgs = [imgs[i].reshape(1, 3, 448, 448) for i in range(len(imgs))] #add transformations
            imgs =[transform(imgs[i]) for i in range(len(imgs))]
        else:
            imgs = [self.transforms_final(imgs[i]) for i in range(len(imgs))]

        extractor = FeatureExtractor(model)
        extractor.to(self.device)

        if self.logger:
            self.logger.info("Embedding generation started")
        features = {}
        for i in tqdm(range(len(imgs))):
            img = imgs[i].to(self.device)
            with torch.no_grad():
                feature = extractor(img)              
            features[i]=feature.cpu().detach().numpy().reshape(-1)
        
        if self.logger:
            self.logger.info("Embedding generation completed")
        self.emb = np.array(list(features.values()))
        return self.emb

    def data_emb_loader(self,data_file,transform=None,batch_size=8,shuffle=False,num_workers=4,logger=None):
        """
        get embedding data from yaml file
        """
        loader = torch.utils.data.DataLoader(data_file, batch_size=batch_size, transform=transform,shuffle=shuffle, num_workers=num_workers)
        return loader


        
