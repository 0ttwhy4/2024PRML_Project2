from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

## Note that: here we provide a basic solution for loading data and transforming data.
## You can directly change it if you find something wrong or not good enough.

def load_data(cfg):
    # data augmentations
    
    train_transforms = [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    
    valid_transforms = [
            transforms.Resize(cfg['val']['input_size']),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    
    transform_cfg = cfg['train']['transform']
    pivot = 1
    if transform_cfg['crop']:
        train_transforms.insert(pivot ,transforms.RandomResizedCrop(cfg['train']['input_size'], scale=(0.2, 1.0)))
        pivot += 1
    if transform_cfg['h_flip']:
        p = transform_cfg['h_flip_p']
        train_transforms.insert(pivot, transforms.RandomHorizontalFlip(p=p))
        pivot += 1
    if transform_cfg['v_flip']:
        p = transform_cfg['v_flip_p']
        train_transforms.insert(pivot, transforms.RandomVerticalFlip(p=p))
        pivot += 1
    if transform_cfg['rot']:
        rot_degrees = transform_cfg['rot_degrees']
        train_transforms.insert(pivot, transforms.RandomRotation(rot_degrees))
        pivot += 1
    if transform_cfg['gaussian_blur']:
        kernel_size = transform_cfg['kernel_size']
        sigma = transform_cfg['sigma']
        p = transform_cfg['gaussian_blur_p']  # 高斯模糊的概率
        train_transforms.insert(pivot, transforms.RandomApply([transforms.GaussianBlur(kernel_size, sigma)], p=p))
        pivot += 1
    
    data_transforms = {
        'train': transforms.Compose(train_transforms),
        'val': transforms.Compose(valid_transforms),
    }
    
    print(data_transforms['train'])
    ## The default dir is for the first task of large-scale deep learning
    ## For other tasks, you may need to modify the data dir or even rewrite some part of 'data.py'
    image_dataset_train = datasets.ImageFolder(os.path.join(cfg['data_dir'], 'train'), data_transforms['train'])
    image_dataset_valid = datasets.ImageFolder(os.path.join(cfg['data_dir'], 'val'), data_transforms['val'])
    
    train_loader = DataLoader(image_dataset_train, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=4)
    valid_loader = DataLoader(image_dataset_valid, batch_size=cfg['val']['batch_size'], shuffle=False, num_workers=4)

    return train_loader, valid_loader
