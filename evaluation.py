import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def compute_knn(backbone, data_loader_train, data_loader_val):
    """
    Get CLS embeddings and use KNN classfier on them.
    
    We load all embeddings in memory and use sklearn. Should be doable.
    
    Parameters
    -----------
    backbone : timm.models.vision_transformer. Vision Transformer
        Vision Transformer whose head is just a identity mapping.
        
    data_loader_train, data_loader_val : torch.utils.data.DataLoader
        Training and validation dataloader that does not apply any
        augmentations. Just casting to tensor and then normalizing.
        
    Returns
    ----------
    val_accuary : float
        Validation accuracy
        
    """
    
    device = next(backbone.parameters()).device
    
    data_loaders = {
        'train': data_loader_train,
        'val' : data_loader_val
    }
    
    lists = {
        'X_train':[],
        'Y_train':[],
        'X_val':[],
        'Y_val':[]
    }
    
    for name, data_loader in data_loaders.items():
        for imgs, y in data_loader:
            imgs = imgs.to(device)
            lists[f'X_{name}'].append(backbone(imgs).detach().cpu().numpy())
            lists[f'Y_{name}'].append(y.detach().cpu().numpy())
            
    arrays = {k:np.concatenate(l) for k,l in lists.items()}
    
    estimator = KNeighborsClassifier()
    estimator.fit(arrays['X_train'], arrays['Y_train'])
    y_val_pred = estimator.predict(arrays['X_val'])
    
    acc = accuracy_score(arrays['Y_val'], y_val_pred)
    
    return acc

def compute_embedding(backbone, data_loader):
    """
    Compute CLS embedding and prepare for Tensorboard.
    
    Parameters
    ------------
    backbone : timm.models.vision_transformer.VisionTransformer
        Vision Transformer. The head should be an identity mapping.
        
    data_loader : torch.utils.data.DataLoader
        Validation dataloader that does not apply any augmentations. Just
        casting to tensor and then normalizing.
        
    Returns
    -----------
    embs : torch.Tensor
        Embedding of shpe '(n_samples, out_dim)'
        
    imgs : torch.Tensor
        Images of shape '(n_samples, 3, height, weight)'
        
    labels : list
        List of strings representing the classes.
    """
    
    device = next(backbone.parameters()).device
    
    embs_l = []
    imgs_l = []
    labels = []
    
    for img, y in data_loader:
        img = img.to(device)
        embs_l.append(backbone(img).detach().cpu())
        imgs_l.append(((img * 0.224) + 0.45).cpu())
        labels.extend([data_loader.dataset.classes[i] for i in y.tolist()])
        
    embs = torch.cat(embs_l, dim = 0)
    imgs = torch.cat(imgs_l, dim = 0)
    
    return embs, imgs, labels