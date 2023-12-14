import torch 
import torch.nn as nn 

def iou(box1,box2,is_pred = True):

    if is_pred == True: 
        ''' 
        BOX = X,Y,W,H
        '''
        x1_min = box1[...,0:1] - box1[...,2:3]/2 
        x1_max = box1[...,0:1] + box1[...,2:3]/2 
        y1_min = box1[...,1:2] - box1[...,3:4]/2 
        y1_max = box1[...,1:2] + box1[...,3:4]/2 
        box1_area = (x1_max - x1_min)*(y1_max - y1_min) 

        x2_min = box2[...,0:1] - box2[...,2:3]/2 
        x2_max = box2[...,0:1] + box2[...,2:3]/2 
        y2_min = box2[...,1:2] - box2[...,3:4]/2 
        y2_max = box2[...,1:2] + box2[...,3:4]/2 
        box2_area = (x2_max - x2_min)*(y2_max - y2_min) 

        xmin = torch.min(x1_min,x2_min) 
        ymin = torch.min(y1_min,y2_min) 
        xmax = torch.max(x1_max,x2_max)
        ymax = torch.max(x2_max,y2_max) 

        intersection = (xmax - xmin).clamp(0) * (ymax - ymin).clamp(0)
        
        union = box1_area + box2_area - intersection 

        iou = intersection/(union + 1e-6) 

        return iou 
    else: 
        ''' 
        BOX = W,H
        '''
        intersection = torch.min(box1[...,0],box2[...,0]) * torch.min(box1[...,1],box2[...,1]) 
        box1_area = box1[...,0] * box1[...,1]
        box2_area = box2[...,0] * box2[...,1] 

        union = box1_area + box2_area - intersection 
        iou = intersection / union 
        return iou 
    
def nms(bboxes):
    pass 


def plot_image():
    pass 

