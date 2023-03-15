import torch
import numpy as np

__all__ = ['extract_sample']

def extract_sample(n_way, n_support, n_query, datax, datay):
    """
  Picks random sample of size n_support+n_querry, for n_way classes
  Args:
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      datax (np.array): dataset of images
      datay (np.array): dataset of labels
  Returns:
      (dict) of:
        (torch.Tensor): sample of images. Size (n_way, n_support+n_query, (dim))
        (int): n_way
        (int): n_support
        (int): n_query
    """
    K = np.random.choice(np.unique(datay), n_way, replace=False)
    sample = np.empty([128],dtype=float)
    for cls in K:
        ##pass a df with event id , taxcode, after_image_id, embeddings
        temp_data = datax[datax['new_taxcode']==cls]
        datax_cls = np.array(temp_data['feat_vec'])

        perm = np.random.permutation(datax_cls)
        sample_cls = perm[:(n_support+n_query)]
        for i in range(len(sample_cls)):
            sample = np.vstack((sample,np.array(sample_cls[i],dtype=float)))
    sample = torch.from_numpy(sample)
    return({
        'images': sample[1:],
        'n_way': n_way,
        'n_support': n_support,
        'n_query': n_query
        })
