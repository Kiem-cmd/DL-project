import torch 
import torch.nn as nn 

from config import config

class CNNBlock(nn.Module):
    def __init__(self,in_channels, out_channels, **kwargs):
        super().__init__() 
        self.conv = nn.Conv2d(in_channels,out_channels,bias = False, **kwargs)
        self.batchNorm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self,x):
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.act(x)
        return x
    
class Yolov1(nn.Module):
    def __init__(self,config,in_channels = 3, *kwargs):
        super().__init__()
        self.config = config 
        self.in_channels = in_channels 
    def create_conv_layer(self): 
        layers = [] 
        in_channels = self.in_channels 

        for layer in self.config: 
            if type(layer) == tuple:
                layers += [
                    CNNBlock(in_channels = in_channels,out_channels=layer[1],kernel_size = layer[0],stride=layer[2],padding = layer[3])
                ]
                in_channels = layer[1]
            elif type(layer) == str: 
                layers += [
                    nn.MaxPool2d(kernel_size=(2,2), stride = (2,2)) 
                ]
            elif type(layer) == list:
                conv1 = layer[0]
                conv2 = layer[1] 
                n = layer[2] 
                for _ in range(n):
                    layers += [
                        CNNBlock(in_channels = in_channels,out_channels=conv1[1],kernel_size = conv1[0],stride = conv1[2], padding = conv1[3])
                    ]
                    layers += [
                        CNNBlock(in_channels = conv1[1],out_channels=conv2[1],kernel_size=conv2[0],stride = conv2[2],padding = conv2[3]) 
                    ]
                    in_channels = conv2[1]
        return nn.Sequential(*layers)
    def create_fc_layer(self,s,b,c): 
        ''' 
        Params: 
            s: Split image to SxS   - int 
            b: num of box           - int 
            c: num of classes       - int 
        
        '''   
        fc_layers = [ 
            nn.Flatten(),
            nn.Linear(1024 * s * s , 496),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1),
            nn.Linear(496, s * s * (c + 5 * b)) 
        ]
        return nn.Sequential(*fc_layers)
    
    def forward(self,x):
        x = self.create_conv_layer()(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.create_fc_layer(7,2,4)(x) 
        return x 

if __name__ == '__main__':
    sample = torch.randn(1,3,448,448) 
    yolo = Yolov1(config = config)(sample)
    print(yolo.shape)
    # In ra các lớp của mô hình
    # for name, module in yolo.named_modules():
    #     print(name, module)
    # print(config)