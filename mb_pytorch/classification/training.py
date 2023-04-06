from ..models.modelloader import ModelLoader
from ..dataloader.loader import DataLoader
import torch
from ..training.train_params import train_helper
import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

yaml_file = '/home/malav/mb_pytorch/scripts/models/loader_y.yaml'
data = DataLoader(yaml_file,logger=None)
data_model = data.data_dict['model']
train_loader, val_loader = data.train_loader, data.val_loader
model = ModelLoader(data_model,logger=None)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss,optimizer,optimizer_dict,scheduler,scheduler_dict = train_helper(data_model)

path_logs = os.path.join(data['data']['work_dir'], 'logs')
writer = SummaryWriter(log_dir=path_logs)


for i in tqdm(range(data_model['model_epochs'])):
    for j,(x,y) in enumerate(train_loader):
        x,y = x.to(device),y.to(device)
        y_pred = model(x)
        loss = loss(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        if i != 0:
            optimizer_dict['lr'] = scheduler.get_last_lr()[0]
            
        optimizer =optimizer(model.parameters(),*optimizer_dict)
        optimizer.step()
        
        if scheduler is not None:
            scheduler(*scheduler_dict).step()
        
        writer.add_scalar('Loss/train', loss.item(), global_step=i)
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, global_step=i)
        
        #get grad cam images
        
        #validation loop
        val_loss = 0
        val_acc = 0
        num_samples = 0
    
        model.eval()
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                output = model(x_val)
                val_loss += loss(output, y_val).item() * x_val.size(0)
                _, preds = torch.max(output, 1)
                val_acc += torch.sum(preds == y_val.data)
                num_samples += x_val.size(0)
            
        val_loss /= num_samples
        val_acc = val_acc.double() / num_samples
    
        writer.add_scalar('Loss/val', val_loss, global_step=i)
        writer.add_scalar('Accuracy/val', val_acc, global_step=i)
    
        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
        
        model.train()
        
        # save best model
        path = os.path.join(data['data']['work_dir'], 'best_model.pth')
        torch.save(best_model, path)