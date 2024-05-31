import torch
import torch.nn as nn
import torch.optim as optim
import data
import models
import os
import time
import argparse
import datetime

## Note that: here we provide a basic solution for training and validation.
## You can directly change it if you find something wrong or not good enough.



def get_parser():
    parser = argparse.ArgumentParser()
    
    ## about data
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_dir', type=str, default="EuroSAT_PRML24/Task_A" ) ## You need to specify the data_dir first
    parser.add_argument('--input_size', type=int, default=64)
    
    ## about training
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    return parser

def train(model, train_loader,optimizer,criterion):
    model.train(True)
    total_loss = 0.0
    total_correct = 0
    cnt = 0
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
            print(f'[TRAIN][{idx}/{len(train_loader)}] Loss={(total_loss/cnt):.4f}')
    epoch_loss = total_loss / len(train_loader.dataset)
    epoch_acc = total_correct.double() / len(train_loader.dataset)
    return epoch_loss, epoch_acc.item()


def valid(model, valid_loader, criterion):
    model.train(False)
    total_loss = 0.0
    total_correct = 0
    for inputs, labels in valid_loader:
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predictions = torch.max(outputs, 1)
        total_loss += loss.item() * inputs.size(0)
        total_correct += torch.sum(predictions == labels.data)
    epoch_loss = total_loss / len(valid_loader.dataset)
    epoch_acc = total_correct.double() / len(valid_loader.dataset)
    return epoch_loss, epoch_acc.item()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = get_parser()
    args = parser.parse_args()
    
    ## model initialization
    model = models.ResNet18(num_classes=10)
    model = model.cuda()

    ## data preparation
    train_loader, valid_loader = data.load_data(args)

    ## optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    ## loss function
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    
    for epoch in range(args.num_epochs):
        start_time = time.time()
        print('*' * 100)
        print(f'Epoch: {epoch}/{args.num_epochs}')
        
        train_loss, train_acc = train(model, train_loader,optimizer,criterion)
        print(f"[TRAIN][Epoch {epoch}] Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        
        valid_loss, valid_acc = valid(model, valid_loader,criterion)
        print(f"[VAL][Epoch {epoch}] Loss={valid_loss:.4f}, Acc={valid_acc:.4f}")
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model
            # save the model if you want to
            # torch.save(best_model, 'best_model.pt')
        print('Epoch Time:', str(datetime.timedelta(seconds=int(time.time() - start_time))))

    print('Best Acc:', best_acc)