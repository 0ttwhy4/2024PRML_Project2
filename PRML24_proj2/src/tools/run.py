import torch
import torch.nn as nn
import os
import time
import datetime
from tools.plot import plot_curve
from tools.build import build_finetune

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

def run_epoch_mixmatch(model, train_loader, optimizer, criterion, logger, scheduler):
    model.train(True)
    total_loss = 0.0
    total_correct = 0
    cnt = 0
    logger.info(f'training size: {train_loader.__len__()} * {train_loader.batch_size} = {train_loader.__len__() * train_loader.batch_size}')
    
    for idx, batch in enumerate(train_loader):
        inputs, labels, src = batch
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels, src)
        _, predictions = torch.max(outputs, 1)
        _, label_pred = torch.max(labels, 1)
        total_correct += torch.sum(predictions == label_pred)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        cnt += inputs.size(0)
        if idx % 50 == 0:
            logger.info(f'[TRAIN][{idx}/{len(train_loader)}] Loss={(total_loss/cnt):.4f}')
    
    epoch_loss = total_loss / len(train_loader.dataset)
    epoch_acc = total_correct.double() / len(train_loader.dataset)
    
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