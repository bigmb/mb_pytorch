from ..models.modelloader import ModelLoader
from ..dataloader.loader import DataLoader
import torch
from ..training.train_params import train_helper
import tqdm

yaml_file = '/home/malav/mb_pytorch/scripts/models/loader_y.yaml'
data = DataLoader(yaml_file,logger=None)
data_model = data.data_dict['model']
train_loader, val_loader = data.train_loader, data.val_loader
model = ModelLoader(data_model,logger=None)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss,optimizer,optimizer_dict,scheduler,scheduler_dict = train_helper(data_model)

for i in tqdm(range(data['model_epochs'])):
    for j,(x,y) in enumerate(train_loader):
        x,y = x.to(device),y.to(device)
        y_pred = model(x)
        loss = loss(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        if i!=0:
            optimizer_dict['lr'] = scheduler.get_lr()[0]
        optimizer = optimizer(model.parameters(),*optimizer_dict).step()
        
        if scheduler is not None:
            scheduler(*scheduler_dict).step()
        