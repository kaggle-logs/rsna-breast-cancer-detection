import cv2
import numpy as np
import pandas as pd
import pydicom

# pytorch
import torch
from torch.utils.data import Dataset

from albumentations import (ToFloat, Normalize, VerticalFlip, HorizontalFlip, Compose, Resize,
                            RandomBrightnessContrast, HueSaturationValue, Blur, GaussNoise,
                            Rotate, RandomResizedCrop, Cutout, ShiftScaleRotate, ToGray)
from albumentations.pytorch import ToTensorV2


class Transform():
    def __init__(self, is_train, **kwargs):
        # Data Augmentation (custom for each dataset type)
        if is_train:
            self.transform = Compose([
                RandomResizedCrop(height=224, width=224),
                ShiftScaleRotate(rotate_limit=90, scale_limit = [0.8, 1.2]),
                HorizontalFlip(p = kwargs["horizontal_flip"]),
                VerticalFlip(p = kwargs["vertical_flip"]),
                Normalize(
                      mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = Compose([
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def get(self):
        return self.transform


class RSNADataset(Dataset):
    
    def __init__(self, dataframe, transform, csv_columns, is_train=True):
        self.dataframe = dataframe
        self.is_train = is_train
        self.csv_columns = csv_columns
        self.transform = transform.get()
            
    def __len__(self):
        return len(self.dataframe)
    
    
    def __getitem__(self, index):
        '''Take each row in batcj at a time.'''
        
        # Select path and read image
        image_path = self.dataframe['path'][index]
        image = pydicom.dcmread(image_path).pixel_array.astype(np.float32)
        
        # For this image also import .csv information
        csv_data = np.array(self.dataframe.iloc[index][self.csv_columns].values, 
                            dtype=np.float32)
        # Apply transforms
        transf_image = self.transform(image=image)['image']
        # Change image from 1 channel (B&W) to 3 channels
        transf_image = np.concatenate([transf_image, transf_image, transf_image], axis=0)
        
        # Return info
        if self.is_train:
            return {"image": transf_image, 
                    "meta": csv_data, 
                    "target": self.dataframe['cancer'][index]}
        else:
            return {"image": transf_image, 
                    "meta": csv_data}


class RSNADatasetPNG(Dataset):
    
    def __init__(self, dataframe, transform, csv_columns, is_train=True):
        self.dataframe = dataframe
        self.is_train = is_train
        self.csv_columns = csv_columns
        self.transform = transform.get()
            
            
    def __len__(self):
        return len(self.dataframe)
    
    
    def __getitem__(self, index):
        '''Take each row in batcj at a time.'''
        
        # Select path and read image
        image_path = self.dataframe['path'][index]
        image = cv2.imread(image_path).astype(np.float32)
        
        # For this image also import .csv information
        csv_data = np.array(self.dataframe.iloc[index][self.csv_columns].values, 
                            dtype=np.float32)
        # Apply transforms
        transf_image = self.transform(image=image)['image']
        
        # Return info
        if self.is_train:
            return {"image": transf_image, 
                    "meta": csv_data, 
                    "target": self.dataframe['cancer'][index]}
        else:
            return {"image": transf_image, 
                    "meta": csv_data}
