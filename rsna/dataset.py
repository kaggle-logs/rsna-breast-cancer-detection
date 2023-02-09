import cv2
import numpy as np
import pandas as pd
import pydicom

# pytorch
import torch
from torch.utils.data import Dataset

# local
import rsna.preprocess as prep


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
    
    def __init__(self, dataframe, transform, csv_columns, has_target=True, image_prep_ver=None):
        self.dataframe = dataframe
        self.has_target = has_target
        self.csv_columns = csv_columns
        self.transform = transform
        self.breast_prep = prep.BreastPreprocessor(image_prep_ver)
            
    def __len__(self):
        return len(self.dataframe)
    
    
    def __getitem__(self, index):
        '''Take each row in batcj at a time.'''
        
        # Select path and read image
        image_path = self.dataframe['path'][index]
        image = prep.load_img(image_path)

        # preprocess
        # - laterality = R なら左右反転 (encodeされているので0/1)
#        if prep.is_flip_side(image):
#            image = np.fliplr(image)
#        # - 胸部のみ抽出
#        image, _ = prep.get_breast_region_2(image)
        image, aux = self.breast_prep.get_breast_region(image)
        
        # For this image also import .csv information
        csv_data = np.array(self.dataframe.iloc[index][self.csv_columns].values, 
                            dtype=np.float32)
        # Apply transforms
        transf_image = self.transform(image=image)['image']
        
        # Return info
        if self.has_target:
            return {"image": transf_image, 
                    "meta": csv_data, 
                    "target": self.dataframe['cancer'][index]}
        else:
            return {"image": transf_image, 
                    "meta": csv_data}
