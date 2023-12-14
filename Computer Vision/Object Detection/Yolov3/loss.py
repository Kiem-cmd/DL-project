import torch
import torch.nn as nn 

class Loss(nn.Module): 
    def __init__(self,num_classes, num_anchors):
        super().__init__() 
        self.num_classes = num_classes 
        self.num_anchors = num_anchors 
        self.mse = nn.MSELoss() 
        self.bce = nn.BCELossWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropy() 
        self.sigmoid = nn.Sigmoid() 


        self.lambda_obj = 1 
        self.lambda_no_obj = 10
        self.lambda_box = 10
        self.lambda_class = 1 

    def forward(self,pred,target,anchor_box):
        obj = target[..., 0]  == 1
        no_obj = target[...,0] == 0 

        ''' 
        CONFIDENT LOSS 
        '''
        no_obj_loss = self.bce(
            (pred[...,0:1][no_obj]),(target[...,0:1][no_obj])
        )
        
        obj_loss = self.bce(self.sigmoid((pred[...,0:1][obj])),target[...,0:1][obj])

        ''' 
        BOX COORDINATE LOSS 

        bx = sigmoid(tx) + cx 
        by = sigmoid(ty) + cy 
        bw = pw.exp(tw) 
        bh = ph.exp(th)
        
        '''
        box_pred = torch.cat([self.sigmoid(pred[...,1:3]),torch.exp(pred[...,3:5]) * anchors], dim = -1)
        

        box_loss = self.mse((pred[...,1:5][obj]),(target[...,1:5])) 



        ''' 
        CLASS LOSS 
        '''
        class_loss = self.bce((pred[...,5:][obj]),target[...,5][obj].long())

        loss = self.lambda_obj*obj_loss + self.lambda_no_obj*no_obj_loss + self.lambda_box* box_loss + self.lambda_class* class_loss 

        return loss 
    