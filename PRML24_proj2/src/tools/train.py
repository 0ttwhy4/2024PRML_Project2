import torch
import torch.nn as nn
import os
import datetime
import time
from tools.run import run_epoch_labeled, run_epoch_mixmatch, valid
from tools.plot import plot_curve
from tools.build import build_tools, build_finetune

def train(model, 
          cfg,
          optimizer,
          criterion,
          train_loader,
          valid_loader,
          scheduler, 
          work_dir,
          logger, 
          save_model,
          task,
          save_name=None
          ):
    
    if save_name is None:
        save_name = '{}_model'.format(task)
    
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
        if task == 'student':
            train_loss, train_acc = run_epoch_mixmatch(model, train_loader, optimizer, criterion, logger, scheduler)
        else:
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
    
    return best_model

def finetune(model, cfg, logger, work_dir, save_model):
    if isinstance(model, str) and os.path.isfile(model):
        model = torch.load(model)
        model = model.cuda()
    optimizer, criterion, train_loader, val_loader, scheduler = build_finetune(model, cfg, logger)
        
    best_acc = 0.0
    ft_loss = []
    ft_acc = []
    ft_val_loss = []
    ft_val_acc = []
    num_epochs = cfg['finetune']['num_epochs']
    
    for epoch in range(1, num_epochs+1):
        start_time = time.time()
        logger.debug('*' * 80)
        logger.info(f'Epoch: {epoch}/{num_epochs}')
            
        epoch_loss, epoch_acc = run_epoch_labeled(model, train_loader, optimizer, criterion, logger, scheduler)
        ft_loss.append(epoch_loss)
        ft_acc.append(epoch_acc)
        
        logger.info(f'[FINETUNE][EPOCH {epoch}] Loss={epoch_loss}, Acc={epoch_acc}')

        val_loss, val_acc = valid(model, val_loader, logger)
        logger.info(f"[VAL][Epoch {epoch}] Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        ft_val_loss.append(val_loss)
        ft_val_acc.append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model
            best_epoch = epoch
            if save_model:
                torch.save(best_model, os.path.join(work_dir, 'finetune_best_model.pt'))
        logger.info('Epoch Time: {}'.format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
    
    logger.debug('*' * 100)
    logger.info('[FINETUNE] Best Epoch: {} | Best Acc: {}'.format(best_epoch, best_acc))
    
    plot_dir = os.path.join(work_dir, 'finetune_curve.png')
    
    plot_curve(ft_loss, ft_acc, ft_val_loss, ft_val_acc, num_epochs, plot_dir)