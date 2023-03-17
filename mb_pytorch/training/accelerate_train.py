#function to train the model with accelerate

from accelerate import Accelerator
import torch


accelerator = Accelerator() #initialize the accelerator

def acc_train(model,config,train_loader, val_loader):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(accelerator.device)
    model,optimizer,lr_scheduler ,train_loader,val_loader = accelerator.prepare(model,config.optimizer,config.lr_scheduler,train_loader,val_loader) #prepare the model, options, train_loader and val_loader for the accelerator
    
    model.train()
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        for batch in train_loader:
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            output = model(batch['image'])
            loss = config.loss_fn(output, batch['label'])
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
        
    model.eval()
    accurate = 0
    num_elems = 0
    for batch in val_loader:
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        batch['image'] = batch['image'].to(device)
        batch['label'] = batch['label'].to(device)
        predictions = model(batch['image'])
        all_predictions, all_targets = accelerator.gather_for_metrics((predictions, batch['label']))
        num_elems += all_predictions.shape[0]
        accurate += all_predictions.long().sum() 
    eval_metric = accurate.item() / num_elems
    # Use accelerator.print to print only on the main process.
    accelerator.print(f"epoch {epoch}: {100 * eval_metric:.2f}")   