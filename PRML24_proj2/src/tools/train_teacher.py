import torch
import torch.nn as nn
import os
import datetime
import time
from torch.nn.functional import softmax
from tools.plot import plot_curve
from tools.build import build_tools
from data.transform import Transform
from data.mixmatch_dataset import UnlabeledDataset
from torch.utils.data import DataLoader

def run_epoch_labeled(model, train_loader, optimizer, criterion, logger, scheduler):
    model.train(True)
    total_loss = 0.0
    total_correct = 0
    cnt = 0
    logger.info(f'training size: {train_loader.__len__()} * {train_loader.batch_size} = {train_loader.__len__() * train_loader.batch_size}')
    
    for idx, batch in enumerate(train_loader):
        inputs, labels = batch
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predictions = torch.max(outputs, 1)
        total_correct += torch.sum(predictions == labels.data)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        cnt += inputs.size(0)
        if idx % 50 == 0:
            logger.info(f'[TRAIN][{idx}/{len(train_loader)}] Loss={(total_loss/cnt):.4f}')
            
    epoch_acc = total_correct.double() / len(train_loader.dataset)
    epoch_loss = total_loss / len(train_loader.dataset)
    scheduler.step()
    
    return epoch_loss, epoch_acc.item()

def valid(model, valid_loader, logger):
    model.train(False)
    total_loss = 0.0
    total_correct = 0
    pred = torch.zeros(10).cuda()
    gt = torch.zeros(10).cuda()

    for inputs, labels in valid_loader:
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        outputs = model(inputs)
        val_criterion = nn.CrossEntropyLoss()
        loss = val_criterion(outputs, labels)
        _, predictions = torch.max(outputs, 1)
        total_loss += loss.item() * inputs.size(0)
        total_correct += torch.sum(predictions == labels.data)
        for label in range(0, 10):
            gt[label] += torch.sum(labels.data == label)
            pred[label] += torch.sum((labels.data == label) & (predictions == label))
            
    acc_rounded = (pred.data / gt.data).detach().cpu().numpy() 
    acc_rounded = [round(acc, 4) for acc in acc_rounded]
    logger.debug('class acc: {}'.format(acc_rounded))
        
    epoch_loss = total_loss / len(valid_loader.dataset)
    epoch_acc = total_correct.double() / len(valid_loader.dataset)
    return epoch_loss, epoch_acc.item()

####################################################

def sharpen(labels, temp):
    sharpen_labels = torch.pow(labels, 1.0/temp)
    sharpen_labels /= (torch.sum(sharpen_labels, dim=1).view(-1, 1))
    return sharpen_labels

def generate_label(model, dataloader, save_label_path, logger, k, temp):
    # before generating labels, first validate the performance of teacher models
    pivot = 0
    label_num = 0
    for inputs, img_name in dataloader:
        pivot += 1
        if pivot % 10 == 0:
            logger.info('processing batch {}/{}'.format(pivot, len(dataloader)))
        inputs = inputs.cuda(non_blocking=True)
        outputs = model(inputs)
        outputs = softmax(outputs, dim=1)
        outputs = outputs.view(outputs.size(0)//k, k, outputs.size(1))
        outputs = torch.sum(outputs, dim=1)
        labels = outputs / k
        labels = sharpen(labels, temp)
        labels = labels.detach().cpu()
        for i, label in enumerate(labels):
            label_num += 1
            file = img_name[i*k] + '.pt'
            save_path = os.path.join(save_label_path, file)
            torch.save(label, save_path)
    logger.info('finish labeling {} images in total!'.format(label_num))
    
#########################################################

def train_teacher(model, 
                  cfg,  
                  work_dir, 
                  logger, 
                  save_model, 
                  save_name='best_model',
                  generate_label=False):
    
    task = 'teacher'
    optimizer, criterion, train_loader, valid_loader, scheduler = build_tools(model, cfg, task)
    tr_loss = []
    tr_acc = []
    val_loss = []
    val_acc = []
    train_epochs = cfg['train']['num_epochs']
    best_epoch = 0
    best_acc = 0.0
    
    for epoch in range(1, train_epochs+1):
        start_time = time.time()
        logger.debug('*' * 80)
        logger.info(f'Epoch: {epoch}/{train_epochs}')
        
        train_loss, train_acc = run_epoch_labeled(model, train_loader, optimizer, criterion, logger, scheduler)
        logger.info(f"[TRAIN][Epoch {epoch}] Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        tr_loss.append(train_loss)
        tr_acc.append(train_acc)
        
        valid_loss, valid_acc = valid(model, valid_loader, logger)
        logger.info(f"[VAL][Epoch {epoch}] Loss={valid_loss:.4f}, Acc={valid_acc:.4f}")
        val_loss.append(valid_loss)
        val_acc.append(valid_acc)
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model
            best_epoch = epoch
            if save_model:
                torch.save(best_model, os.path.join(work_dir, '{}.pt'.format(save_name)))
        logger.info('Epoch Time: {}'.format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
        
    logger.debug('*' * 100)
    logger.info('Best Epoch: {} | Best Acc: {}'.format(best_epoch, best_acc))
        
    plot_dir = os.path.join(work_dir, 'curve.png')
    
    plot_curve(tr_loss, tr_acc, val_loss, val_acc, train_epochs, plot_dir)
    
    #########################################################
    if generate_label:
        data_dir = cfg['data_dir']
        
        transform = Transform()
        unlabel_transform = transform(cfg['assign_label']['transform'])
        
        unlabeled_path = os.path.join(data_dir, 'train_unlabeled', 'data')
        save_label_path = os.path.join(data_dir, 'train_unlabeled', 'pseudo_label_regenerate')
        if not os.path.exists(save_label_path):
            os.mkdir(save_label_path)
        unlabeled_data = UnlabeledDataset(transform=unlabel_transform, 
                                        data_dir=unlabeled_path, 
                                        K=2)
        
        dataloader = DataLoader(unlabeled_data, 
                                batch_size=cfg['assign_label']['batch_size'], 
                                shuffle=False, # DO NOT SHUFFLE
                                num_workers=4)
        
        generate_label(best_model, dataloader, '/home/stu6/EuroSAT_PRML24/Task_B/train_unlabeled/pseudo_label_regenerate', logger, 2, 5.0)