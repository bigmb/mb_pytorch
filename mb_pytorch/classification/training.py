from ..models.modelloader import ModelLoader
from ..dataloader.loader import DataLoader
import torch

yaml_file = '/home/malav/mb_pytorch/scripts/models/loader_y.yaml'
data = DataLoader(yaml_file,logger=None)
data_model = data.data_dict['model']
train_loader, val_loader = data.train_loader, data.val_loader
model = ModelLoader(data_model,logger=None)

loss,optimizer,scheduler = 