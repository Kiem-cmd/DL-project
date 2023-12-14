''' 
* Localzization loss 

* Confident loss 

* Classification losss

'''

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
 
class Yolo_Loss(nn.Module):
    def __init__(self, S , B , C): 
        ''' 
        
        '''
        self.s = S 
        self.b = B 
        self.c = C 
        self.mse = nn.MSELoss(reduction="sum")
    def forward(self,pred,target): 
        pred = pred.reshape(-1,self.s,self.s, self.c + self.b * 5)

        iou_b1 = iou(pred[...,],target[...,])
        iou_b2 = iou(pred[...,],target[...,]) 

        ious = torch.cat([iou_b1.unsqueeze(0),iou_b2.unsqueeze(0)], dim = 0)
        box_loss = 

        loss = \
            self.lambda_coord * box_loss    \
            + object_loss                    \
            + self.lambda_noobj * noobj_loss  \
            + classifi_loss
        