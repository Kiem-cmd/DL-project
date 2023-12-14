import os 
import cv2 
import torch 
import torch.nn as nn 
import pandas as pd 
from torch.utils.data import Dataset
from helps import iou
import pandas as pd 

class Fruit_Dataset(Dataset): 
    def __init__(self,path_csv_label,image_dir,anchors): 
        self.image_size = 416 
        self.grid_size = [13,26,52] 
        self.num_classes = 3 
        self.num_anchors = 9 
        self.num_anchors_per_scale = 3 
        self.ignore_iou_thresh = 0.5 

        self.dataframe_label = pd.read_csv(path_csv_label)
        self.image_dir = image_dir 
        
        self.anchors = torch.tensor(
            anchors[0] + anchors[1] + anchors[2]
        ) 
        


    def __len__(self):
        return self.dataframe_label['name'].nunique()
    def __getitem__(self,index): 
        list_name = self.dataframe_label['name'].unique() 
        sample = list_name[index] 
        label = self.dataframe_label[self.dataframe_label['name'] == sample]
        image = cv2.imread(os.path.join(self.image_dir,f"{sample}.jpg")) 


        bboxes = [] 
        for i,row in label.iterrows(): 
            x,y,width,height,class_label = row['x'],row['y'],row['width'],row['height'], row['label']
            bboxes.append((x,y,width,height,class_label)) 
        target = [torch.zeros((self.num_anchors_per_scale,s,s,6)) for s in self.grid_size] 
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]),self.anchors,is_pred = False) 
            anchor_indexs = iou_anchors.argsort(descending = True,dim = 0) 

            x,y,w,h,c = box 

            has_anchor = [False] * 3 
            for anchor_idx in anchor_indexs:
                scale_idx = anchor_idx // self.num_anchors_per_scale 
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale 

                s = self.grid_size[scale_idx] 

                i,j  = int(s * y), int(s * x) 
                anchor_taken = target[scale_idx][anchor_on_scale,i,j,0] 

                if not anchor_taken and not has_anchor[scale_idx]:
                    target[scale_idx][anchor_on_scale,i,j,0] = 1 
                    x_cell, y_cell = s * x - j, s * y - i  
                    width_cell, height_cell = w*s, h*s 
                    box_coord = torch.tensor(
                        [x_cell,y_cell,width_cell,height_cell]
                    )
                    target[scale_idx][anchor_on_scale,i,j,1:5] = box_coord 
                    target[scale_idx][anchor_on_scale,i,j,5] = int(class_label) 
                    has_anchor[scale_idx] = True 
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    target[scale_idx][anchor_on_scale,i,j,0] = -1 
        return image,tuple(target) 
    
if __name__ == '__main__':
    ANCHORS = [ 
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
    ]

    label = '../Yolov3/data/train/label.csv' 
    image_dir = '../Yolov3/data/train/image/' 
    dataset = Fruit_Dataset(label,image_dir,ANCHORS) 

    loader = torch.utils.data.DataLoader( 
    dataset=dataset, 
    batch_size=1, 
    shuffle=True, 
    ) 
    x, y = next(iter(loader)) 
    print("DONE")