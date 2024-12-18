##Class for generating embeddings for the given data

import numpy as np
from tqdm import tqdm

__all__ = ['EmbeddingGenerator']


class EmbeddingGenerator:
    def __init__(self, model,dataloader,layer='avgpool',layer_name=None,device='cpu', logger=None) -> None:
        """
        Initialize the EmbeddingGenerator class
        Input:
            model: model for which embeddings are to be generated
            dataloader: dataloader for the data
            layer: layer from which embeddings are to be generated. Default: avgpool. if None, then it will look for the layer_name
            layer_name: name of the layer. Only used if layer is None. Default: None. Eg: 'self.model.layer4[2].downsample[1]'
            device: device on which the model is to be run
            logger: logger instance
        """
        self.model = model
        self.dataloader = dataloader
        self.layer_name = layer_name
        self.layer = layer
        self.device = device
        self.logger = logger

    def generate_emb(self):
        """
        Generate embeddings for the given data
        Input:
            data: data for which embeddings are to be generated (numpy array)
        Output:
            emb: embeddings for the given data
        """
        self.model.to(self.device)
        self.model.eval()
        
        features_blobs = []
        def get_hook(module, input, output):
            N,C,H,W = output.shape
            output = output.reshape(N,C,-1)
            features_blobs.append(output.data.cpu().numpy())

        if self.layer:
            fea = self.model._modules.get(self.layer).register_forward_hook(get_hook)
        else:
            fea = eval(self.layer_name).register_forward_hook(get_hook)

        if self.logger:
            self.logger.info("Embedding generation started")
            self.logger.info("length of data: {}".format(len(self.dataloader.dataset)))

        for i,i_dat in tqdm(enumerate(self.dataloader),total=len(self.dataloader.dataset)):
            _ = self.model(i_dat['image'].to(self.device))

        self.emb= np.concatenate(features_blobs)
        self.emb = self.emb.reshape(self.emb.shape[0],self.emb.shape[1])

        fea.remove()
        del self.model
        if self.logger:
            self.logger.info("Embedding generation completed")
        return self.emb
