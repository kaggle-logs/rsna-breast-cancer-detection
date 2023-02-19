import cv2
import numpy as np
import pandas as pd
import pydicom

# pytorch
import torch
from torch.utils.data import Dataset, Sampler

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
        # デフォルトではグレースケールで画像を読み込んでいる
        image_path = self.dataframe['path'][index]
        image = prep.load_img(image_path, color = "rgb")

        # preprocess
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
                    "target": self.dataframe['cancer'][index],
                    "prediction_id"  : str(self.dataframe['patient_id'][index]) + "_" + str(self.dataframe["laterality"][index])
                    }
        else:
            return {"image": transf_image, 
                    "meta": csv_data, 
                    "prediction_id"  : str(self.dataframe['patient_id'][index]) + "_" + str(self.dataframe["laterality"][index])
                    }


class BalanceSampler(Sampler):
    """
    neg:pos を bs-1:1 の割合で混ぜるためのSampler

    - pos_index, neg_index の作成
    - length でバッチ分割したときのもの * バッチ数 で全データ数を作成（正確な全データではない、落としているデータもある）
    - pos, neg の indexをシャッフルする
    - neg_index, [:length] で全数を取得、reshape(-1, r) で 全データ//31 の個数分のバッチができる
    - pos_index, 適当にバッチの個数（バッチサイズではなく、バッチが何個あるか）とってくる
    - これらを concat すると、31:1の割合で pos があるバッチが出来上がる（常にposがあるということになる）
    """

    def __init__(self, dataset, ratio=8):
        self.r = ratio-1
        self.dataset = dataset
        self.pos_index = np.where(dataset.dataframe.cancer>0)[0]
        self.neg_index = np.where(dataset.dataframe.cancer==0)[0]

        self.length = self.r*int(np.floor(len(self.neg_index)/self.r))

    def __iter__(self):
        pos_index = self.pos_index.copy()
        neg_index = self.neg_index.copy()
        np.random.shuffle(pos_index)
        np.random.shuffle(neg_index)

        neg_index = neg_index[:self.length].reshape(-1,self.r)
        pos_index = np.random.choice(pos_index, self.length//self.r).reshape(-1,1)

        index = np.concatenate([pos_index,neg_index],-1).reshape(-1)
        return iter(index)

    def __len__(self):
        return self.length