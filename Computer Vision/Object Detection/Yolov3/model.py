import torch 
import torch.nn as nn 
from config import config

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,**kwargs) 
        self.batch = nn.BatchNorm2d(out_channels) 
        self.act = nn.LeakyReLU(0.1) 
    def forward(self,x): 
        x = self.conv(x) 
        x = self.batch(x) 
        x = self.act(x) 
        return x 
    
class ResBlock(nn.Module):
    def __init__(self,channels,n): 
        super().__init__() 
        self.n = n 
        self.channels = channels 

        layers = [] 
        for _ in range(self.n): 
            layers += [
                nn.Sequential(
                    CNNBlock(in_channels = channels, out_channels = channels * 2, kernel_size = 1),
                    CNNBlock(in_channels = channels*2,out_channels = channels,kernel_size = 3, padding = 1 )
                )
            ]
        self.layers = nn.ModuleList(layers) 

    def forward(self, x): 
        for layer in self.layers: 
            res = x 
            x = layer(x) 
            x =  x + res 
        return x 

class Predict(nn.Module): 
    def __init__(self,channels,num_classes,num_box): 
        super().__init__() 
        self.num_classes = num_classes 
        self.num_box = num_box

        self.pred = nn.Sequential(
            CNNBlock(in_channels= channels, out_channels= channels * 2,kernel_size = 3, padding = 1) ,
            nn.Conv2d(channels*2, (5 + self.num_classes) * self.num_box, kernel_size = 1)
        )
    def forward(self,x): 
        ''' 
        x : (B,num_box,s,s,(5+c))
        '''

        out = self.pred(x) 
        out = out.view(x.shape[0],self.num_box,5+self.num_classes,x.shape[2],x.shape[3])
        out = out.permute(0,1,3,4,2)
        return out 
    


class Yolov3(nn.Module):
    def __init__(self,config,in_channels = 3 , num_classes = 3, num_box = 3):
        super().__init__() 
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_box = num_box
        self.config  = config
        layers = []
        for i,layer in enumerate(config): 
            if type(layer) == list: 
                layers += [
                    CNNBlock(in_channels=layer[0],out_channels= layer[1],kernel_size = layer[2],stride = layer[3],padding = layer[4])
                ]
            elif type(layer) == tuple: 
                layers += [
                    ResBlock(channels= layer[0], n = layer[1])
                ]
            elif type(layer) == int: 
                layers += [
                    Predict(channels=layer,num_classes= self.num_classes,num_box= self.num_box)
                ]
            elif type(layer) == str: 
                layers += [
                    nn.Upsample(scale_factor = 2)
                ]
        self.layers = nn.ModuleList(layers) 

    def forward(self,x):
        output = []
        route = []
        for i,layer in enumerate(self.layers): 
            if isinstance(layer,Predict): 
                output.append(layer(x)) 
                continue 
            x = layer(x)
        
            if isinstance(layer,ResBlock) and layer.n == 8: 
                route.append(x) 
            elif isinstance(layer,nn.Upsample):
                x = torch.cat([x,route[-1]],dim = 1) 
                route.pop() 
        return output  
        

if __name__ == '__main__':
    sample = torch.rand(1,3,416,416) 
    yolo = Yolov3(config) 
    print(yolo(sample)[0].shape)
    print(yolo(sample)[1].shape)
    print(yolo(sample)[2].shape)