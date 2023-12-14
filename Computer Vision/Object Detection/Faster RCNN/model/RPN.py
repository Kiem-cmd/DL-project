import torch 
import torch.nn as nn 
from torch.nn import functional as F 

class RPN(nn.Module):
    def __init__(self,in_channels = 512, mid_channel= 512):
        super().__init__() 

        