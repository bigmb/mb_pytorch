from ..models.modelloader import ModelLoader
from ..dataloader.loader import DataLoader
import torch
from ..training.train_params import train_helper
import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from mb_utils.src.logging import logger

yaml_file = '/home/malav/mb_pytorch/scripts/models/loader_y.yaml'
data = DataLoader(yaml_file,logger=None)
data_model = data.data_dict['model']
train_loader, val_loader,_,_ = data.data_load(logger=None)
model = ModelLoader(data_model,logger=None)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_attr,optimizer_attr,optimizer_dict,scheduler_attr,scheduler_dict = train_helper(data_model)

#optimizer_dict['lr'] = scheduler_attr.get_last_lr()[0]
optimizer =optimizer_attr(model.parameters(),**optimizer_dict)
scheduler = scheduler_attr(optimizer,**scheduler_dict)

best_val_loss = float('inf')

path_logs = os.path.join(data['data']['work_dir'], 'logs')
writer = SummaryWriter(log_dir=path_logs)

logger = logger


for i in tqdm.tqdm(range(data_model['model_epochs'])):
    ##train loop
    model.train()
    train_loss = 0
    logger.info('Training Started')
    for j,(x,y) in enumerate(train_loader):
        x,y = x.to(device),y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        current_loss = loss_attr()(y_pred,y)
        current_loss.backward()    
        optimizer.step()
        train_loss += current_loss.item()
        logger.info(f'Epoch {i+1} - Batch {j+1} - Train Loss: {current_loss.item()}')

    avg_train_loss = train_loss / len(train_loader)
    logger.info(f'Epoch {i+1} - Train Loss: {avg_train_loss}')
    
    writer.add_scalar('Loss/train', loss_attr().item(), global_step=i)
    
    if scheduler is not None:
        scheduler.step()
    
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
            val_loss += loss_attr()(output, y_val).item() * x_val.size(0)
            _, preds = torch.max(output, 1)
            val_acc += torch.sum(preds == y_val.data)
            num_samples += x_val.size(0)
            logger.info(f'Epoch {i+1} - Batch {j+1} - Val Loss: {val_loss}')
            
        val_loss /= num_samples
        val_acc = val_acc.double() / num_samples
        logger.info(f'Epoch {i+1} - Val Loss: {val_loss}', f'Epoch {i+1} - Val Accuracy: {val_acc}')
    
    writer.add_scalar('Loss/val', val_loss, global_step=i)
    writer.add_scalar('Accuracy/val', val_acc, global_step=i)
    
    # save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model.state_dict()

        path = os.path.join(data['data']['work_dir'], 'best_model.pth')
        torch.save(best_model, path)
        logger.info(f'Epoch {i+1} - Best Model Saved')
        
    model.train()
        
