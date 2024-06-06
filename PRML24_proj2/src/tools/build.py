import models
import torch
from tools import data
import torch.nn as nn
import torch.optim as optim

optimizers = {'SGD': optim.SGD,
             'Adam': optim.Adam,
             'AdamW': optim.AdamW,
             'RMSprop': optim.RMSprop}

def build_model(cfg):
    model_params = cfg['model']['params']
    model = models.ResNet18(num_classes=10, **model_params)
    model = model.cuda()
    return model

def build_optim(model, cfg):
    optim_name = cfg['optimizer']['type']
    params = cfg['optimizer']['param']
    optimizer = optimizers[optim_name](model.parameters(), **params)
    return optimizer

def from_cfg(cfg):
    model = build_model(cfg)
    optimizer = build_optim(model, cfg)
    criterion = nn.CrossEntropyLoss()
    train_loader, valid_loader = data.load_labeled(cfg)
    return model, optimizer, criterion, train_loader, valid_loader
    