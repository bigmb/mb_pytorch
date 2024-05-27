import torch
from typing import Optional
import tqdm
import os
from mb_utils.src.logging import logger
import numpy as np
from ..utils.viewer import gradcam_viewer,create_img_grid,plot_classes_pred
from mb_pytorch.models.modelloader import ModelLoader
from mb_pytorch.training.train_params import train_helper

__all__ = ['detection_train_loop']

def detection_train_loop( k_yaml: dict,scheduler: Optional[object] =None,writer: Optional[object] =None,
                              logger: Optional[object] =None,gradcam: Optional[object] =None,
                              gradcam_rgb: str =False,device: str ='cpu'):
    """
    Function to train the model
    Args:
        k_yaml: data dictionary YAML of DataLoader
        scheduler: scheduler
        writer: tensorboard writer
        logger: logger
        gradcam: gradcam layers to be visulized
        device: default is cpu
    output:
        None
    """
    
    if logger:
        logger.info('Training loop Starting')
    k_data = k_yaml.data_dict['data']
    data_model = k_yaml.data_dict['model']
    model_data_load = ModelLoader(k_yaml.data_dict['model'])
    model =  model_data_load.get_model()
    
    if logger:
        logger.info('Model Loaded')
    
    train_loader,val_loader,_,_ = k_yaml.data_load()
    loss_attr,optimizer_attr,optimizer_dict,scheduler_attr,scheduler_dict = train_helper(data_model) 
    optimizer = optimizer_attr(model.parameters(),**optimizer_dict)
    if scheduler is not None:
        scheduler = scheduler_attr(optimizer,**scheduler_dict)

    if logger:
        logger.info('Optimizer and Scheduler Loaded')
        logger.info(f'Loss: {loss_attr}')
        logger.info(f'Optimizer: {optimizer}')
        logger.info(f'Scheduler: {scheduler}')
    
    model.to(device)

    for i in tqdm.tqdm(range(data_model['model_epochs'])):
        
        ##train loop
        
        model.train(True)
        train_loss = 0
        
        if logger:
            logger.info('Training Started')
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            temp_dict = {}
            final_list = []
            temp_dict['boxes'] = targets[1][:]
            temp_dict['labels'] = targets[0][:]
            final_list = [temp_dict]
            targets = [{k: v.to(device) for k, v in t.items()} for t in final_list]
            
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            losses.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            train_loss += losses.item()
            if logger:
                logger.info(f'Epoch {i+1} - Batch {batch_idx+1} - Train Loss: {losses.item()}')
        
        avg_train_loss = train_loss / len(train_loader)
        if logger:
            logger.info(f'Epoch {i+1} - Train Loss: {avg_train_loss}')
            logger.info(f"lr = {optimizer.param_groups[0]['lr']}")

        model.train(False)
    
        if writer is not None:
            writer.add_graph(model, x)
            writer.add_scalar('Loss/train', avg_train_loss, global_step=i)
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, global_step=i)
            
            x = x.to('cpu')
            y = y.to('cpu')
            x_grad = x[0,:]
            x_grad = x_grad.unsqueeze(0)
            #y_grad = y[0].to('cpu')
            
            create_img_grid(x,y,writer,global_step=i)

            ##gradcam       
            if gradcam is not None:
                use_cuda=False
                if device.type != 'cpu':
                    use_cuda = True
                for cam_layers in gradcam:
                    grad_img = gradcam_viewer(cam_layers,model,x_grad,gradcam_rgb=gradcam_rgb,use_cuda=use_cuda)
                    if grad_img is not None:
                        grad_img = np.transpose(grad_img,(2,0,1))
                        writer.add_image(f'Gradcam training/{cam_layers}',grad_img,global_step=i)
                    if grad_img is None and logger:
                        logger.info(f'Gradcam not supported for {cam_layers}')            
                        
        ## Validation loop
        val_loss = 0
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                images = list(image.to(device) for image in images)
                temp_dict = {}
                final_list = []
                temp_dict['boxes'] = targets[1][:]
                temp_dict['labels'] = targets[0][:]
                final_list = [temp_dict]
                targets = [{k: v.to(device) for k, v in t.items()} for t in final_list]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                val_loss += losses.item() * len(images)
                if logger: 
                    logger.info(f'Epoch {i+1} - Batch {batch_idx+1} - Val Loss: {losses.item()}')
            
            avg_val_loss = val_loss / len(val_loader.dataset)
            
            if logger:
                logger.info(f'Epoch {i+1} - Avg Val Loss: {avg_val_loss:.3f}')
    
        if writer is not None:
            writer.add_scalar('Loss/val', avg_val_loss, global_step=i)
            
            # Grad-CAM visualization for validation images
            if gradcam is not None:
                use_cuda = device != 'cpu'
                for cam_layers in gradcam:
                    grad_img = gradcam_viewer(cam_layers, model, images[0].unsqueeze(0), gradcam_rgb=gradcam_rgb, use_cuda=use_cuda)
                    if grad_img is not None:
                        grad_img = np.transpose(grad_img, (2, 0, 1))
                        writer.add_image(f'Gradcam/{cam_layers}', grad_img, global_step=i)
                    if grad_img is None and logger:
                        logger.info(f'Gradcam not supported for {cam_layers}')   
   
        # Save best model
        if i == 0:
            best_val_loss = float('inf')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

            path = os.path.join(k_data['work_dir'], 'best_model.pth')
            torch.save(best_model, path)
            if logger:
                logger.info(f'Epoch {i+1} - Best Model Saved')