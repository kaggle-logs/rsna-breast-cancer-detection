import os
import torch
from dataclasses import dataclass

if os.path.exists("/kaggle") : 
    PLATFORM = "kaggle"
else : 
    PLATFORM = "local"

if PLATFORM == "kaggle" : 
    INPUT_PATH = "/kaggle/input/rsna-breast-cancer-detection/"
elif PLATFORM == "local" : 
    INPUT_PATH = "/Users/ktakeda/workspace/kaggle/rsna-breast-cancer-detection/input/rsna-breast-cancer-detection/"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TPU = False