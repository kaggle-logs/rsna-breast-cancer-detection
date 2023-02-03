import os
import torch
from dataclasses import dataclass
import torch_xla
import torch_xla.core.xla_model as xm

if os.path.exists("/kaggle") : 
    PLATFORM = "kaggle"
else : 
    PLATFORM = "local"

if PLATFORM == "kaggle" : 
    INPUT_PATH = "/kaggle/input/rsna-breast-cancer-detection/"
elif PLATFORM == "local" : 
    INPUT_PATH = "./input/rsna-breast-cancer-detection/"

TPU = False
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif xm.xla_device():
    DEVICE = xm.xla_device()
    TPU = True
else:
    DEVICE = 'cpu'

@dataclass
class TrainConfig:
    fold : int
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
