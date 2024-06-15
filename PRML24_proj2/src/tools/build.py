import models
import torch.nn as nn
import torch.optim as optim
import os
import importlib.util
from data import load_data, dataset
from data.transform import Transform
from data.dataset import LabeledDataset, GuessLabelDataset, mix_match
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from torchvision.datasets import ImageFolder
from tools.loss import MixMatchLoss

optimizers = {'SGD': optim.SGD,
             'Adam': optim.Adam,
             'AdamW': optim.AdamW,
             'RMSprop': optim.RMSprop}

model_names = {'resnet18': models.ResNet18,
               'resnet50': models.ResNet50}

def load_cfg(config_file_path):
    spec = importlib.util.spec_from_file_location("config", config_file_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    
    config_dict = {}
    for key in dir(config):
        if not key.startswith("__"):
            config_dict[key] = getattr(config, key)
    return config_dict['config']

def build_model(cfg, mode):
    
    model_name = cfg[mode]['model']['type']
    model_params = cfg[mode]['model']['params']
    model = model_names[model_name]
    model = model(num_classes=10, **model_params)
    model = model.cuda()
    return model

def build_optim(model, cfg, mode):
    optim_name = cfg[mode]['optimizer']['type']
    params = cfg[mode]['optimizer']['param']
    optimizer = optimizers[optim_name](model.parameters(), **params)
    return optimizer

def build_dataset(cfg, mode):
    data_dir = cfg['data_dir']
    transform = Transform()
    
    val_dataset = ImageFolder(os.path.join(data_dir, 'val'), transform(cfg['val']['transform']))
    
    if mode == 'teacher':
        transform = transform(cfg['labeled_dataset']['transform'])
        labeled_dataset = ImageFolder(os.path.join(data_dir, 'train_labeled'), transform)
        return labeled_dataset, val_dataset
    elif mode == 'student':
        labeled_transform = transform(cfg['labeled_dataset']['transform'])
        labeled_dataset = LabeledDataset(os.path.join(data_dir, 'train_labeled'), labeled_transform)
        guess_labeled_transform = transform(cfg['unlabeled_dataset']['transform'])
        guess_labeled_dataset = GuessLabelDataset(os.path.join(data_dir, 'train_unlabeled'), guess_labeled_transform, K=cfg['unlabeled_dataset']['K'])
        return mix_match(labeled_dataset, guess_labeled_dataset, alpha = cfg['alpha']), val_dataset
    else:
        raise ValueError('Invalid task type. --mode should be either *teacher* or *student*, but got {}'.format(mode))

def load_data(cfg, mode):
    # TIPS: before you use, you should print the transformation config to make sure the data is correctly augmented
    train_set, val_set = build_dataset(cfg, mode)
    bs = cfg[mode]['train']['batch_size']
    train_loader = DataLoader(dataset=train_set, 
                              batch_size=bs,
                              shuffle=True,
                              num_workers=4)
    val_loader = DataLoader(val_set, batch_size=cfg['val']['batch_size'], shuffle=False, num_workers=4)
    return train_loader, val_loader

def from_cfg(cfg, mode):
    model = build_model(cfg, mode)
    optimizer = build_optim(model, cfg, mode)
    if mode == "teacher":
        criterion = nn.CrossEntropyLoss()
    else: 
        criterion = MixMatchLoss(10, 100)
    
    train_loader, valid_loader = load_data(cfg, mode)
    return model, optimizer, criterion, train_loader, valid_loader
    