import torch 
import torch.nn as nn
from torch.nn import functional as F 
from RPN import RPN
class fasterRCNN(nn.Module):
    def __init__(self):
        self.rpn = RPN()
    def forward(self):
        batch_size = img.size(0) 

    

