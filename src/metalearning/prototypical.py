import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from ..utils.dist import euclidean_dist

__all__ = ['ProtoNet']


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def load_protonet_conv(**kwargs):
    x_dim=kwargs['x_dim']
    y_dim=kwargs['y_dim']
    z_dim=kwargs['z_dim']

    def conv_block(in_ch,out_ch):
        return nn.Sequential(
                nn.Conv2d(in_ch,out_ch,1,stride=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2))
    encoder=nn.Sequential(
            conv_block(x_dim,y_dim),
            conv_block(y_dim,z_dim),
            Flatten())
    return PrototypicalNetworks(encoder)

class PrototypicalNetworks(nn.Module):
    def __init__(self,encoder):
        """
        Args:
            encoder : CNN encoding the images in sample
            n_way (int): number of classes in a classification task
            n_support (int): number of labeled examples per class in the support set
            n_query (int): number of labeled examples per class in the query set
        """
        super(PrototypicalNetworks, self).__init__()
        self.encoder = encoder

    def set_forward_loss(self, sample):
        """
        Computes loss, accuracy and output for classification task
        Args:
            sample (torch.Tensor): shape (n_way, n_support+n_query, (dim)) 
        Returns:
            torch.Tensor: shape(2), loss, accuracy and y_hat
        """
        sample_images = sample['images'].cuda()
        n_way = sample['n_way']
        n_support = sample['n_support']
        n_query = sample['n_query']

        x_support = sample_images[:, :n_support]
        x_query = sample_images[:, n_support:]

        #target indices are 0 ... n_way-1
        target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        target_inds = target_inds.cuda()

        #Gettig images in V stack
        z = sample_images
        z_dim=z.size(-1)
        z_proto = z[:n_way*n_support].view(n_way, n_support, z_dim).mean(1)
        z_query = z[n_way*n_support:]

        #compute distances
        dists = euclidean_dist(z_query, z_proto)

        #compute probabilities
        log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'y_hat': y_hat}
                                                 