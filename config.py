import torch
from dataclasses import dataclass

# INPUT_PATH = "./input/rsna-breast-cancer-detection/"
INPUT_PATH = "/kaggle/input/rsna-breast-cancer-detection/"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class TrainConfig:
    fold : int
    patience : int 
    epochs : int
    num_workers : int
    lr : float
    wd : float
    lr_patience : float
    lr_factor : float
    batch_size_1 : int
    batch_size_2 : int
    vertical_flip : float
    horizontal_flip : float
    output_size : int
    csv_columns : list
