from abc import abstractmethod

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

class Meter(object):
    def __init__(self, name):
        self.sum = 0.0
        self.count = 0.0
        self.last = 0.0
        self.name = name
        
    def __call__(self, val):
        self.last = val
        self.sum += self.last
        self.count += 1
        
    def mean(self):
        return self.sum / max(1, self.count)
    
    def value(self):
        return self.last
    
    def reset(self):
        self.sum = 0.0
        self.count = 0.0
        
    def __repr__(self):
        return f'{self.name}: {self.value():.4f}'
        

class InterfaceTrainingVisualizer(object):
        
    @abstractmethod
    def add_scalars(self, scalars: dict, step: int, prefix: str = '') -> None:
        raise NotImplemented
            
    @abstractmethod
    def add_images(self, images: dict, step: int , prefix: str = '') -> None:
        raise NotImplemented
        
    @abstractmethod
    def close(self) -> None:
        raise NotImplemented
        

class SegmentationVisualizer(InterfaceTrainingVisualizer):
    def __init__(self, logdir: str =None) -> None:
        self.writer = SummaryWriter(logdir)
        
    def add_scalars(self, scalars: dict, step: int, prefix: str = '') -> None:
        for k,v in scalars.items():
            self.writer.add_scalar(prefix+k, v, step)
            
    def add_images(self, images: dict, step: int, prefix: str = '') -> None:
        for k,v in images.items():
            assert isinstance(v, torch.Tensor)
            grid = torchvision.utils.make_grid(v)
            if len(grid.size()) == 3:
                grid = grid.unsqueeze(0)
            self.writer.add_images(prefix+k, grid, step)
            
        if 'gt_masks' in images and 'images' in images:
            grid = torchvision.utils.make_grid(images['gt_masks'] * images['images'])
            if len(grid.size()) == 3:
                grid = grid.unsqueeze(0)
            self.writer.add_images(prefix+'gt_composition', grid, step)
            
        if 'masks' in images and 'images' in images:
            grid = torchvision.utils.make_grid(images['masks'] * images['images'])
            if len(grid.size()) == 3:
                grid = grid.unsqueeze(0)
            self.writer.add_images(prefix+'composition', grid, step)
    
    def close(self):
        self.writer.close()