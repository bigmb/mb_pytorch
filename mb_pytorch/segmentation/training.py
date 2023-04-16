#Training loop for segmentaion data

from ..models.modelloader import ModelLoader
from ..dataloader.loader import DataLoader
import torch
from ..training.train_params import train_helper
import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from mb_utils.src.logging import logger
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from ..utils.viewer import new_show_cam_on_image


__all__ = ['segmentation_train_loop']



def segmentation_train_loop(k_data,data_model,model,train_loader,val_loader,loss_attr,optimizer,scheduler=None,writer=None,logger=None,gradcam=None,gradcam_rgb=False,device='cpu'):
    """
    Function to train the model
    Args:
        data: data dictionary YAML of DataLoader
        data_model: model parameters - data.data_dict['model']
        model: model to be trained
        train_loader: train dataloader
        val_loader: validation dataloader
        loss_attr: loss function
        optimizer: optimizer
        scheduler: scheduler
        writer: tensorboard writer
        logger: logger
        gradcam: gradcam layers to be visulized
        device: default is cpu
    output:
        None
    """

    for i in tqdm.tqdm(range(data_model['model_epochs'])):
        ##train loop
        model.train()
        train_loss = 0
        if logger:
            logger.info('Training Started')
        for j,(x,y) in enumerate(train_loader):
            x,y = x.to(device),y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            current_loss = loss_attr()(y_pred,y)
            current_loss.backward()    
            optimizer.step()
            train_loss += current_loss.item()
            if logger:
                logger.info(f'Epoch {i+1} - Batch {j+1} - Train Loss: {current_loss.item()}')
                    #get grad cam images
        
            if gradcam and writer is not None:
                x_grad = x.to('cpu')
                for cam_layers in gradcam:
                    if logger:
                        logger.info(f'Gradcam for layer {cam_layers} started')
                    with GradCAM(model=model,target_layers=[cam_layers],use_cuda=False) as cm: 
                        cr = cm(input_tensor=x_grad)[0,:]        
                        #cr2 = np.reshape(cr,[1,cr.shape[0],cr.shape[1]])
                        #if cr2.max() == 0 and cr2.min() == 0:
                        #    cr2 = cr2 + 1
                        #print(cr.shape)
                    cam_img = new_show_cam_on_image(x_grad[0].numpy(),cr,use_rgb=gradcam_rgb)
                    writer.add_image(f'Gradcam/{cam_layers}',cam_img,global_step=i)
        
        avg_train_loss = train_loss / len(train_loader)
        if logger:
            logger.info(f'Epoch {i+1} - Train Loss: {avg_train_loss}')
    
        if writer is not None:
            writer.add_scalar('Loss/train', avg_train_loss, global_step=i)
    
        if scheduler is not None:
            scheduler.step()
    
        if writer is not None:
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, global_step=i)
        
        #validation loop
        val_loss = 0
        val_acc = 0
        #num_samples = 0
    
        model.eval()
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                output = model(x_val)
                val_loss += loss_attr()(output, y_val).item() * x_val.size(0)
                _, preds = torch.max(output, 1)
                val_acc += torch.sum(preds == y_val.data)
                #num_samples += x_val.size(0)
                if logger: 
                    logger.info(f'Epoch {i+1} - Batch {j+1} - Val Loss: {val_loss}')
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_acc/len(val_loader)
            #val_loss /= num_samples
            #val_acc = val_acc / num_samples
            if logger:
                logger.info(f'Epoch {i+1} -Avg Val Loss: {avg_val_loss:.3f}', f'Epoch {i+1} - Val Accuracy: {val_acc:.3f}')
    
        if writer is not None:
            writer.add_scalar('Loss/val', val_loss, global_step=i)
            writer.add_scalar('Accuracy/val', val_acc, global_step=i)
    
        # save best model
        if i == 0:
            best_val_loss = float('inf')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

            path = os.path.join(k_data['work_dir'], 'best_model.pth')
            torch.save(best_model, path)
            if logger:
                logger.info(f'Epoch {i+1} - Best Model Saved')
        
        model.train()
        