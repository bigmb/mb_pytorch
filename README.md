# mb_pytorch
malav pytorch files 



[![Close inactive issues](https://github.com/bigmb/mb_pytorch/actions/workflows/stale.yml/badge.svg)](https://github.com/bigmb/mb_pytorch/actions/workflows/stale.yml)

## Applications
Segmentation
Classification 
Meta Learning

## Scripts


## Visualization
Gradio
Tensorboard 

# Data loader
loading data 
```
    from src.dataloader.loader import DataLoader
    from mb_utils.src.logging import logger
    k = DataLoader('./scripts/loader_y.yaml')
    out1,out2,o1,o2 =k.data_load(logger=logger)
```