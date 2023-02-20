import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, LongTensor
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision.models import resnet34, resnet50

import timm


def get_model(model_type, model_name, pretrained) : 

    if model_type == "resnet" : 
        return ResNeXt(model_name, pretrained)
    elif model_type == "effnet" : 
        return EfficientNet(model_name, pretrained)
    else : 
        raise NotImplementedError


class ResNet50Network(nn.Module):
    
    def __init__(self, output_size, num_columns, is_train = True):
        super().__init__()
        self.num_columns = num_columns
        self.output_size = output_size
        
        # Define Feature part (IMAGE)
        # if is_train = True, the pretrained weights will be downloaded.
        # this competition cannot connect to internet, then the pretrained weights
        # shouldnt be downloaded at the submit stage.
        self.features = resnet50(pretrained=is_train) # 1000 neurons out
        # (metadata)
        self.csv = nn.Sequential(nn.Linear(self.num_columns, 500),
                                 nn.BatchNorm1d(500),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2))
        
        # Define Classification part
        self.classification = nn.Linear(1000 + 500, output_size)
        
        
    def forward(self, image, meta, prints=False):
        if prints: print('Input Image shape:', image.shape, '\n'+
                         'Input metadata shape:', meta.shape)
        
        # Image CNN
        image = self.features(image)
        if prints: print('Features Image shape:', image.shape)
        
        # CSV FNN
        meta = self.csv(meta)
        if prints: print('Meta Data:', meta.shape)
            
        # Concatenate layers from image with layers from csv_data
        image_meta_data = torch.cat((image, meta), dim=1)
        if prints: print('Concatenated Data:', image_meta_data.shape)
        
        # CLASSIF
        out = self.classification(image_meta_data)
        if prints: print('Out shape:', out.shape)
        
        return out


class ResNeXt(nn.Module):
    def __init__(self, model_name="seresnext50_32x4d", pretrained=False):
        super().__init__()

        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.dense1 = nn.Linear(self.model.fc.out_features, 500)
        self.dense2 = nn.Linear(500, 1)

    def forward(self, img, meta):
        x = self.model(img)
        # x = F.adaptive_avg_pool2d(x,1)
        # x = torch.flatten(x,1,3)
        x = self.dense1(x)
        x = self.dense2(x)

        return x


class EfficientNet(nn.Module):

    def __init__(self, model_name="efficientnet_b0", pretrained=False, out_dim=1, only_head=False):
        super().__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=3)
        self.forward_features = self.backbone.forward_features
        self.dense = nn.Linear(1792, out_dim)
        
        # ヘッドだけ学習させるなら
        if only_head : 
            for param in self.backbone.parameters():
                param.requires_grad = False
        
    def forward(self, x, meta, verbose=False):
        x = self.forward_features(x.float())
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1, 3)
        x = self.dense(x)
        x = x.reshape(-1)

        return x