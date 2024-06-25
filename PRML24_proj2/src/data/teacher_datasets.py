from torchvision import datasets
from torch.utils.data import Subset
import torch

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

class LabeledDataset(datasets.ImageFolder):
    
    def __init__(self, data_path, transform):
        super().__init__(root=data_path, transform=transform)
        
    def __getitem__(self, index: int):
        src_set = 0
        data, label_id = super().__getitem__(index)
        label = torch.zeros(10)
        label[label_id] = 1
        return data, label, src_set
    
def build_binary_dataset(target_class, dataset, ratio):
    '''
    ratio: the ratio of positive:negative samples
    '''
    positive_indices = []
    negative_indices = []
    for i, label in enumerate(dataset.targets):
        if label == target_class:
            positive_indices.append(i)
        else:
            negative_indices.append(i)
    shuffle_indices = torch.randperm(len(negative_indices))
    shuffle_negative = negative_indices[shuffle_indices]
    shuffle_negative = shuffle_negative[:ratio * len(positive_indices)]
    binary_indices = positive_indices + shuffle_negative
    return Subset(dataset, binary_indices)