import os
import cv2 
import torch 
import torch.nn as nn 
from torchvision import transforms
from torchvision.models import resnet50 
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
from torchvision.transforms import ToTensor
from torch.nn import CrossEntropyLoss 
from torch.nn import MSELoss 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import argparse 
from tqdm.auto import tqdm 
import xml.etree.ElementTree as ET








class Pet(Dataset):
    def __init__(self,image_path,label_path,transform = None):
        self.paths = [os.path.join(image_path,i) for i in os.listdir(image_path)]
        self.label_path = label_path 
        self.transform = transform 
    def load_image(self,idx):
        image_path = self.paths[idx] 
        return cv2.imread(image_path) 
    def load_label(self,idx): 
        label_path = self.label_path + "/" + self.paths[idx].split("/")[-1][:-4] + ".xml"
        tree = ET.parse(label_path)
        root = tree.getroot()
        bnd_ = root.find('object').find('bndbox')
        class_ = root.find('object').find('name')
        x1 = int(bnd_.find('xmin').text)
        y1 = int(bnd_.find('ymin').text)
        x2 = int(bnd_.find('xmax').text)
        y2 = int(bnd_.find('ymax').text)
        bbox = (x1,y1,x2,y2) 
        if class_ == "dog": 
            label = 0
        else: label = 1
        return (label,bbox)

    def __len__(self):
        return len(self.paths) 
    def __getitem__(self,idx): 
        image = self.load_image(idx)
        label = self.load_label(idx) 
        if self.transform:
            return self.transform(image),label
        else:
            return image,label








class DERT(nn.Module):
    def __init__(self,hidden_dim,num_heads,num_encoder_layers,num_decoder_layers,num_classes):
        super().__init__() 
        self.backbone = nn.Sequential()
        self.conv = nn.Conv(2048,hidden_dim,1) 
        self.transformer = nn.Tranformer(hidden_dim, num_heads, num_encoder_layers, num_decoder_layers) 
        self.linear_class = nn.Linear(hidden_dim,num_classes + 1) 
        self.linear_bbox = nn.Linear(hidden_dim,4)
        self.row_embed = nn.Parameter(torch.rand(50,hidden_dim // 2)) 
        self.col_embed = nn.Parameter(torch.rand(50,hidden_dim // 2)) 

        self.query_pos = nn.Parameter(torch.rand(100,hidden_dim)) 

    
    
    
    def forward(self,inputs):
        x = self.backbone(inputs) 
        h = self.conv(x) 
        H, W = h.shape[-2:] 
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H,1,1), 
            self.row_emb[:H].unsqueeze(1).repear(1,W,1) 
        ], dim = -1).flatten(0,1).unsqueeze() 

        t = self.transformer(pos + 0.1*x.flatten(2).permute(2,0,1), self.query_pos.unsqueeze(1)).transpose(0,1) 
        return (self.linear_class(t),self.linear_bbox(h).sigmoid()) 
    

class Loss_fn(nn.Module):
    def __init__(self,alpha,beta):
        super().__init__()
        self.class_loss = nn.CrossEntropyLoss() 
        self.bbox_loss = nn.MSELoss() 
        self.alpha = alpha 
        self.beta = beta 
    def forward(self,y_pred,y): 
        class_l = self.class_loss(y_pred[0],y[0]) 
        bbox_l = self.bbox_loss(y_pred[1], y[1]) 
        return self.alpha * class_l + self.beta * bbox_l
    
def train_1_step(model,dataloader,loss_fn,optimizer):
    ## 1. Train mode model 
    model.train() 
    ## 2. Train loss & Train acc of 1 Step 
    train_loss  = 0 
    for batch, (x,y) in enumerate(dataloader): 
        ## GPU 
        x, y = x.to(device), y.to(device) 
        ## Forward 
        y_pred = model(x) 
        ## Cal loss
        loss = loss_fn(y_pred,y) 
        train_loss += loss.item()  
        ## Optimize 
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
    train_loss = train_loss / len(dataloader) 
    return train_loss 



def train(model,train_loader,test_loader,epochs, loss_fn):  
    optimizer = torch.optim.Adam() 
    loss = ... 
    for epoch in tqdm(range(epochs)): 
        train_loss  = train_1_step(model = model,
                                    dataloader = train_loader,
                                    loss = loss_fn,
                                    optimizer = optimizer) 
        test_loss = test_1_step(model = model,
                                dataloader = test_loader,
                                loss = loss_fn) 
        print(f"Epoch: {epoch} | Train_Loss: {train_loss} | Test_loss: {test_loss}")





def predict(predict,num_pred = 10):
    pass 

def display(img,pred):
    pass 


if '__name__' == '__main__': 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser() 
    parser.add_argument("--hidden_dim", type = int, default = 128) 
    parser.add_argument("--num_heads",type = int, default = 8)
    parser.add_argument("--num_encoder_layers0", type = int, defaule = 6) 
    parser.add_argument('--num_decoder_layers', type = int, defaule = 6) 
    parser.add_argument("--epochs", type = int, default = 10) 
    parser.add_argument("--image_path", type = str, default = "/home/kiem/Github/DL-project/Computer Vision/Object Detection/Data/OxfordPet/images")
    parser.add_argument("--label_path", type = str, default = "/home/kiem/Github/DL-project/Computer Vision/Object Detection/Data/OxfordPet/labels")


    print("Building Dataset...........") 
    train_transforms = transforms.Compose([
                            transforms.Resize((64, 64)),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.ToTensor()
                        ])
    test_transforms = transforms.Compose([
                            transforms.Resize((64, 64)),
                            transforms.ToTensor()
                        ])
    train_dataset = Pet(image_path,label_path, transform = train_transforms)
    ## test_dataset = Pet()
    print("Done")
    print("--------------------------------------")
    print("Define training and test Dataloader") 
    train_loader =  DataLoader(dataset = train_dataset,
                               batch_size = 16, 
                               num_workers = 0,
                               shuffle = True) 
    # test_loader =  DataLoader(dataset = test_dataset,
    #                            batch_size = 16, 
    #                            num_workers = 0,
    #                            shuffle = True) 
    print(f"DataLoader: {train_loader, test_loader}") 
    print(f"Length of train dataloader: {len(train_loader)}") 
    print(f"Length of test dataloader: {len(test_loader)}") 
    print("---------------------------------------") 
    print("Building model ........................")
    model = DERT() 
    train(train_loader,test_loader,epochs) 


 