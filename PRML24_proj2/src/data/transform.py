from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class Transform():
    
    def __init__(self):
        self.transforms = [transforms.ToTensor(), 
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        
    
    def __call__(self, transform_cfg):
        pivot = 1
        if transform_cfg['crop']:
            self.transforms.insert(pivot, transforms.RandomResizedCrop(transform_cfg['input_size'], scale=(0.2, 1.0)))
            pivot += 1
        else:
            self.transforms.insert(pivot, transforms.Resize(transform_cfg['input_size']))
            pivot += 1
        if transform_cfg['h_flip']:
            p = transform_cfg['h_flip_p']
            self.transforms.insert(pivot, transforms.RandomHorizontalFlip(p=p))
        if transform_cfg['v_flip']:
            p = transform_cfg['v_flip_p']
            self.transforms.insert(pivot, transforms.RandomVerticalFlip(p=p))
            pivot += 1
        if transform_cfg['gaussian_blur']:
            kernel_size = transform_cfg['kernel_size']
            sigma = transform_cfg['sigma']
            p = transform_cfg['gaussian_blur_p']  # Probability of applying Gaussian blur
            self.transforms.insert(pivot, transforms.RandomApply([transforms.GaussianBlur(kernel_size, sigma)], p=p))
            pivot += 1
        return transforms.Compose(self.transforms)