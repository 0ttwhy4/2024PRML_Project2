import torch
import os
import time
import argparse
import datetime
from tools.logger import get_logger
import json
from tools.plot import plot_curve
from tools.build import from_cfg
from tools.config import config as cfg

## Note that: here we provide a basic solution for training and validation.
## You can directly change it if you find something wrong or not good enough.

def get_parser():
    parser = argparse.ArgumentParser()
    default_cfg = '/home/stu6/2024PRML_Project2/PRML24_proj2/src/tools/config.py'
    parser.add_argument('--cfg', type=str, default=default_cfg)
    parser.add_argument('--save_model', action='store_true')
    return parser

def train(model, train_loader,optimizer,criterion, logger):
    model.train(True)
    total_loss = 0.0
    total_correct = 0
    cnt = 0
    static = torch.zeros((10,)).cuda()
    logger.info(f'training size: {train_loader.__len__()} * {train_loader.batch_size} = {train_loader.__len__() * train_loader.batch_size}')
    
    for idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predictions = torch.max(outputs, 1)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_correct += torch.sum(predictions == labels.data)
        cnt += inputs.size(0)
        if idx % 50 == 0:
            logger.info(f'[TRAIN][{idx}/{len(train_loader)}] Loss={(total_loss/cnt):.4f}')
    epoch_loss = total_loss / len(train_loader.dataset)
    epoch_acc = total_correct.double() / len(train_loader.dataset)
    return epoch_loss, epoch_acc.item()


def valid(model, valid_loader, criterion, logger):
    model.train(False)
    total_loss = 0.0
    total_correct = 0
    pred = torch.zeros(10).cuda()
    gt = torch.zeros(10).cuda()

    for inputs, labels in valid_loader:
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
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


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = get_parser()
    args = parser.parse_args()
    
    work_dir = cfg['work_dir']
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    model, optimizer, criterion, train_loader, valid_loader = from_cfg(cfg)
    logger = get_logger(os.path.join(work_dir, 'logging.log'))
    logger.debug("model config:\n %s", json.dumps(cfg, indent=4))
    
    best_acc = 0.0
    
    tr_loss = []
    tr_acc = []
    val_loss = []
    val_acc = []
    num_epochs = cfg['train']['num_epochs']
    
    for epoch in range(1, num_epochs+1):
        start_time = time.time()
        logger.debug('*' * 80)
        logger.info(f'Epoch: {epoch}/{num_epochs}')
        
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, logger)
        logger.info(f"[TRAIN][Epoch {epoch}] Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        tr_loss.append(train_loss)
        tr_acc.append(train_acc)
        
        valid_loss, valid_acc = valid(model, valid_loader, criterion, logger)
        logger.info(f"[VAL][Epoch {epoch}] Loss={valid_loss:.4f}, Acc={valid_acc:.4f}")
        val_loss.append(valid_loss)
        val_acc.append(valid_acc)
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model
            if args.save_model:
                torch.save(best_model, os.path.join(work_dir, 'best_model.pt'))
        logger.info('Epoch Time: {}'.format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
        
    plot_dir = os.path.join(work_dir, 'curve.png')    
    plot_curve(tr_loss, tr_acc, val_loss, val_acc, num_epochs, plot_dir)

    logger.debug('*' * 100)
    logger.info('Best Acc: {}'.format(best_acc))