import torch.nn as nn
from lovasz import lovasz_hinge


class EdgeLoss(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.dilate = nn.Conv2d(1,1, kernel_size, padding=kernel_size//2, bias=False)
        self.dilate.weight.requires_grad = False
        self.dilate.weight[...] = 1.0
        
        self.erode = nn.Conv2d(1,1, kernel_size, padding=kernel_size//2, bias=False, padding_mode='replicate')
        self.erode.weight.requires_grad = False
        self.erode.weight[...] = 1.0
        
    def forward(self, y_head, y):
        y_edge = y.clone()
        y_edge[self.dilate(y)<0.5] = 255
        y_edge[self.erode(1-y)<0.5] = 255

        loss = lovasz_hinge(y_head, y, per_image=False)
        loss += 0.5 * lovasz_hinge(y_head, y_edge, ignore=255, per_image=False)
        return loss
