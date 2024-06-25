import models
import torch
import torch.nn as nn
import torch.optim as optim
import os
import importlib.util
from data import load_data
from data.transform import Transform
from data.teacher_datasets import LabeledDataset
from data.mixmatch_dataset import GuessLabelDataset, mixup
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

def build_teacher_model(model_cfg, model_path=None):
    
    if isinstance(model_path, str) and os.path.isfile(model_path):
        model = torch.load(model_path)
        model = model.cuda()
        return model
    
    model_name = model_cfg['type']
    model_params = model_cfg['params']
    
    if model_cfg['binary_class']:
        model = []
        for _ in range(10):
            bn_model = model_names[model_name]
            bn_model = model(num_classes=1, **model_params)
            bn_model = bn_model.cuda()
            model.append(bn_model)
        return model
    else:
        model = model_names[model_name]
        model = model(num_classes=10, **model_params)
        model = model.cuda()
        return model

def build_student_model(model_cfg, model_path=None):
    if isinstance(model_path, str) and os.path.isfile(model_path):
        model = torch.load(model_path)
        model = model.cuda()
        return model
    model_name = model_cfg['type']
    model_params = model_cfg['params']
    model = model_names[model_name]
    model = model(num_classes=10, **model_params)
    model = model.cuda()
    return model   

def build_optim(model, **optim_param):
    optim_name = optim_param['type']
    params = optim_param['param']
    optimizer = optimizers[optim_name](model.parameters(), **params)
    return optimizer

def build_labeled_dataset(data_dir, transform_cfg):
    transform = Transform()(transform_cfg)
    labeled_dataset = LabeledDataset(data_dir, transform)
    return labeled_dataset

def build_guesslabeled_dataset(data_dir, transform_cfg, k):
    pseudo_labeled_transform = Transform()(transform_cfg)
    pseudo_labeled_dataset = GuessLabelDataset(data_dir, pseudo_labeled_transform, k)
    return pseudo_labeled_dataset

def build_mixmatch_dataset(data_root, labeled_transform_cfg, unlabeled_transform_cfg, k, alpha):

    labeled_dir = os.path.join(data_root, 'train_labeled')
    labeled_dataset = build_labeled_dataset(labeled_dir, labeled_transform_cfg)
    
    guesslabel_dir = os.path.join(data_root, 'train_unlabeled')
    guess_labeled_dataset = build_guesslabeled_dataset(guesslabel_dir, unlabeled_transform_cfg, k)
    
    return mixup(labeled_dataset, guess_labeled_dataset, alpha)
    

def build_dataset(cfg, task):
    val_cfg = cfg['val']
    val_data_dir = os.path.join(cfg['data_dir'], 'val')
    val_dataset = ImageFolder(val_data_dir, Transform()(val_cfg['transform']))
    
    if task == 'teacher':
        data_dir = os.path.join(cfg['data_dir'], 'train_labeled')
        transform_cfg = cfg['train']['dataset']['transform']
        train_dataset = ImageFolder(data_dir, Transform()(transform_cfg))
        return train_dataset, val_dataset
    
    elif task == 'student':
        data_root = cfg['data_dir']
        trainset_cfg = cfg['train']['dataset']
        labeled_transform_cfg = trainset_cfg['labeled_transform']
        unlabeled_transform_cfg = trainset_cfg['unlabeled_transform']
        k = trainset_cfg['K']
        alpha = trainset_cfg['alpha']
        mixmatch_dataset = build_mixmatch_dataset(data_root, labeled_transform_cfg, unlabeled_transform_cfg, k, alpha)
        return mixmatch_dataset, val_dataset
        
    else:
        raise ValueError('Invalid task type. --mode should be either *teacher* or *student*, but got {}'.format(task))
    

def load_data(cfg, task):
    # TIPS: before you use, you should print the transformation config to make sure the data is correctly augmented
    train_set, val_set = build_dataset(cfg, task)
    batch_size = cfg['train']['batch_size']
    train_loader = DataLoader(dataset=train_set, 
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)
    val_loader = DataLoader(val_set, batch_size=cfg['val']['batch_size'], shuffle=False, num_workers=4)
    return train_loader, val_loader

def build_criterion(task, cfg):
    if task == "teacher":
        if cfg['model']['binary_class']:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
    else: 
        criterion = MixMatchLoss(10, cfg['train']['lambda_u'])
    return criterion

def build_tools(model, cfg, task):
    optim_param = cfg['train']['optimizer']
    optimizer = build_optim(model, **optim_param)
    criterion = build_criterion(task, cfg)
    train_loader, valid_loader = load_data(cfg, task)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **cfg['train']['scheduler'])
    return optimizer, criterion, train_loader, valid_loader, scheduler

def build_finetune(model, cfg):
    finetune_param = cfg['finetune']
    optim_type = finetune_param['optimizer']['type']
    optim_param = finetune_param['optimizer']['param']
    optimizer = optimizers[optim_type](filter(lambda p: p.requires_grad, model.parameters()), **optim_param)
    
    criterion = nn.CrossEntropyLoss()
    
    train_transform = Transform()(finetune_param['dataset']['transform'])
    finetune_dataset = ImageFolder(os.path.join(cfg['data_dir'], 'train_labeled'), train_transform)
    finetune_loader = DataLoader(dataset=finetune_dataset,
                                 batch_size=finetune_param['batch_size'],
                                 shuffle=True,
                                 num_workers=4)
    
    val_transform = Transform()(cfg['val']['transform'])
    val_dataset = ImageFolder(os.path.join(cfg['data_dir'], 'val'), val_transform)
    val_loader = DataLoader(val_dataset, batch_size=cfg['val']['batch_size'], shuffle=False, num_workers=4)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **cfg['finetune']['scheduler'])
    
    return optimizer, criterion, finetune_loader, val_loader, scheduler
    