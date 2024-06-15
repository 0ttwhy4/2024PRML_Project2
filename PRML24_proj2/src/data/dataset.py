from torchvision import datasets
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import os
import torch
from scipy.stats import beta

class2label = {'AnnualCrop': 0, 
               'Forest': 1, 
               'HerbaceousVegetation': 2, 
               'Highway': 3, 
               'Industrial': 4, 
               'Pasture': 5, 
               'PermanentCrop': 6, 
               'Residential': 7, 
               'River': 8, 
               'SeaLake': 9}


class UnlabeledDataset(Dataset):
    
    '''
    NOTE: Do NOT shuffle this dataset when labeling
    '''
    
    def __init__(self, transform, data_dir, K=1):
        self.K = K 
        self.path = data_dir
        self.transform = transform
    
    def __len__(self):
        return self.K * len(os.listdir(self.path))
    
    def __getitem__(self, idx):
        idx_ = idx // self.K
        img = os.listdir(self.path)[idx_]
        img_name = img.split('.')[0]
        img = os.path.join(self.path, img)
        img = Image.open(img).convert('RGB')
        img = self.transform(img)
        return img, img_name
    


class LabeledDataset(datasets.ImageFolder):
    
    def __init__(self, data_path, transform):
        super().__init__(root=data_path, transform=transform)
        
    def __getitem__(self, index: int):
        src_set = 0
        data, label_id = super().__getitem__(index)
        label = torch.zeros(10)
        label[label_id] = 1 
        return data, label, src_set 
        
        
        
class GuessLabelDataset(Dataset):
    
    def __init__(self, data_dir, transform, K=2):
        self.data_dir = data_dir
        self.data_path = os.path.join(self.data_dir, 'data')
        self.label_path = os.path.join(self.data_dir, 'pseudo_label')
        self.K = K
        self.transform = transform

    def __len__(self):
        return self.K * len(os.listdir(self.data_path))
    
    def __getitem__(self, idx):
        '''
        Here we choose to load each data and perform transformation in __getitem__ method instead of preload the data into a list
        '''
        aug_id = idx // self.K
        data_names = os.listdir(self.data_path)
        data_name = data_names[aug_id]
        img_path = os.path.join(self.data_path, data_name)
        data_name = data_name.split('.')[0]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        # The label is saved in .pt format
        label_file = os.path.join(self.label_path, data_name+'.pt')
        label = torch.load(label_file)
        src_set = 1
        return img, label, src_set
    
    
        
class MixMatchDataset(Dataset):
    
    def __init__(self, gt_dataset, concat_dataset, indices, alpha):
        assert gt_dataset.__len__() == len(indices)
        self.origin = gt_dataset
        self.concat = concat_dataset
        self.indices = indices
        self.beta = beta(alpha, alpha)
        
    def mix(self, ori_X, ori_y, concat_X, concat_y):
        lam = self.beta.rvs(1)[0]
        lam = lam if lam > 0.5 else 1-lam
        X = lam * ori_X + (1-lam) * concat_X
        y = lam * ori_y + (1-lam) * concat_y
        return X, y
    
    def __len__(self):
        return self.origin.__len__()        
    
    def __getitem__(self, index):
        ori_X, ori_y, src_set = self.origin.__getitem__(index) 
        concat_X, concat_y, _ = self.concat.__getitem__(self.indices[index])
        X, y = self.mix(ori_X, ori_y, concat_X, concat_y)
        # transformation has been done in __getitem__ of the two separate sets
        return X, y, src_set # data is mainly composed of original data
    
    
        
def mix_match(X, U, alpha):
    W = ConcatDataset([X, U])
    len_W = W.__len__()
    indices = torch.randperm(len_W)
    X_indices = indices[:X.__len__()]
    U_indices = indices[X.__len__():]
    X_ = MixMatchDataset(X, W, X_indices, alpha)
    U_ = MixMatchDataset(U, W, U_indices, alpha)
    mix_match_set = ConcatDataset([X_, U_])
    return mix_match_set