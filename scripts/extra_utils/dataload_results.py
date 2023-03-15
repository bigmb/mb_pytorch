#!/usr/bin/python3

## script to load results from a basic dataloading script

import os
import sys
from mb_pytorch.dataloader.loader import DataLoader
from mb_utils.src.logging import logger
import argparse



def main():
    data = args.file

    k =DataLoader('./scripts/loader_y.yaml',logger=logger) #load the data

    if logger:
        logger.info("Data loaded")
        logger.info("Data params: {}".format(k.load_data_params))

    out1,out2,o1,o2 =k.data_load(logger=logger)
    if logger:
        #logger.info("self.trainset [0] = {}".format(o1[0]))
        logger.info("self.trainset keys = {}".format(o1[0].keys()))
        logger.info("self.trainset shape = {}".format(o1[0]['image'].shape))

    for i in out1:
        if logger:
            #logger.info("self.trainloader [0] = {}".format(i['image']))
            logger.info("self.trainloader keys = {}".format(i.keys()))
            logger.info("self.trainloader shape = {}".format(i['image'].shape))
            if 'label' in i.keys():
                logger.info("self.trainloader label = {}".format(i['label']))
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',type=str,help='path to yaml data file',default='./scripts/loader_y.yaml')
    args = parser.parse_args()
    main()
