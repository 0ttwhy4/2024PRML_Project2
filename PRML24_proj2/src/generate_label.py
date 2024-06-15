import torch
from torchvision.datasets import ImageFolder
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from config.ssl_config import config as cfg
from data.dataset import UnlabeledDataset
from data.transform import Transform
from tools.logger import get_logger
import os
import json


# generate pseudo label for the unlabeled dataset using pretrained teacher models

def load_models(model_path):
    model_list = os.listdir(model_path)
    models = []
    for model in model_list:
        model_file = os.path.join(model_path, model)
        model = torch.load(model_file)
        model = model.to('cuda')
        model.train(False)
        models.append(model)
    return models
    
def sharpen(labels, T):
    sharpen_labels = torch.pow(labels, 1.0/T)
    sharpen_labels /= (torch.sum(sharpen_labels, dim=1).view(-1, 1))
    return sharpen_labels

def val_teacher(models: list, val_loader, criterion, logger):
    total_loss = 0.0
    total_correct = 0
    for inputs, labels in val_loader:
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        outputs = 0
        for model in models:
            outputs += model(inputs)
        outputs /= len(models)
        loss = criterion(outputs, labels)
        _, predictions = torch.max(outputs, 1)
        total_loss += loss.item() * inputs.size(0)
        total_correct += torch.sum(predictions == labels.data)
    
    val_loss = total_loss / len(val_loader.dataset)
    val_acc = total_correct.double() / len(val_loader)
    logger.info('val loss: {} | acc: {%.2f}'.format(val_loss, val_acc))

def generate_label(dataloader, save_label_path, logger):
    # before generating labels, first validate the performance of teacher models
    pivot = 0
    label_num = 0
    for inputs, img_name in dataloader:
        pivot += 1
        if pivot % 10 == 0:
            logger.info('processing batch {}/{}'.format(pivot, len(dataloader)))
        inputs = inputs.cuda(non_blocking=True)
        outputs = 0
        for model in models:
            outputs += softmax(model(inputs), dim=1)
        outputs = outputs.view(outputs.size(0)//K, K, outputs.size(1))
        outputs = torch.sum(outputs, dim=1)
        labels = outputs / (len(models) * K)
        labels = sharpen(labels, T, logger)
        labels = labels.detach().cpu()
        for i, label in enumerate(labels):
            label_num += 1
            file = img_name[i*K] + '.pt'
            save_path = os.path.join(save_label_path, file)
            torch.save(label, save_path)
    logger.info('finish labeling {} images in total!'.format(label_num))
        
if __name__ == '__main__':
    
    work_dir = cfg['work_dir']
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    logger = get_logger(os.path.join(work_dir, 'logging.log'))
    
    label_config = cfg['assign_label']
    unlabeled_config = cfg['unlabeled_dataset']
    data_dir = cfg['data_dir']
    
    K = unlabeled_config['K']
    T = label_config['T']
    
    models = load_models(label_config['teacher_file'])
    logger.info('loading {} models for ensemble labeling'.format(len(models)))
    
    transform = Transform()
    unlabel_transform = transform(unlabeled_config['transform'])
    
    unlabeled_path = os.path.join(data_dir, 'train_unlabeled', 'data')
    save_label_path = os.path.join(data_dir, 'train_unlabeled', 'pseudo_label')
    unlabeled_data = UnlabeledDataset(transform=unlabel_transform, 
                                      data_dir=unlabeled_path, 
                                      K=K)
    
    dataloader = DataLoader(unlabeled_data, 
                            batch_size=label_config['batch_size'], 
                            shuffle=False, # DO NOT SHUFFLE
                            num_workers=4)
    
    logger.debug('labeling config:\n %s', json.dumps(label_config, indent=4))
    logger.debug('unlabeled set config:\n %s', json.dumps(unlabeled_config, indent=4))
    
    criterion = torch.nn.CrossEntropyLoss()
    val_path = os.path.join(data_dir, 'val')
    val_transform = transform(cfg['val']['transform'])
    val_dataset = ImageFolder(val_path, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    val_teacher(models, val_loader, criterion, logger)
    
    generate_label(dataloader, save_label_path, logger)