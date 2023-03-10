import cv2
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import os
from tqdm import tqdm
import dicomsdl
import pydicom

import numpy as np
import random
import torch
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from rsna.config import INPUT_PATH, DEVICE, PLATFORM

def fix_seed(SEED=1993):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(SEED)


def load_data(f_name, custom_path = None, external=False):
    # Add path column
    
    # Read in Data
    data = pd.read_csv(f"{INPUT_PATH}/{f_name}.csv")
    data["patient_id"] = data["patient_id"].apply(str)
    data["image_id"] = data["image_id"].apply(str)

    if PLATFORM == "kaggle" : 
        if custom_path : 
            if external :
                data["path"] = custom_path + "/" + data["patient_id"] + "_" + data["image_id"] + ".png"
            else:
                data["path"] = custom_path + "/" + data["patient_id"] + "/" + data["image_id"] + ".png"
        else :
            data["path"] = INPUT_PATH + "/" + f_name + "_images/" + data["patient_id"] + "/" + data["image_id"] + ".dcm"
    elif PLATFORM == "local" : 
        # modify the path for local test
        #
        if custom_path : 
            data["path"] = custom_path + "/" + data["patient_id"] + "/" + data["image_id"] + ".png"
        else :
            data["path"] = f"/Users/ktakeda/workspace/kaggle/rsna-breast-cancer-detection/input/rsna-breast-cancer-detection/train_images/10006/462822612.dcm"

    return data
    
    


def data_to_device(data, is_train = True):
    if is_train:
        image, metadata, targets, prediction_id = data.values()
        return image.to(DEVICE), metadata.to(DEVICE), targets.to(DEVICE), prediction_id
    else:
        image, metadata, prediction_id = data.values()
        return image.to(DEVICE), metadata.to(DEVICE), prediction_id

def dicom2png(fname, PNG_SIZE=(256,256), mode="dicomsdl"):

    if mode == "pydicom" : 
        dicom = pydicom.dcmread(fname)
        data = dicom.pixel_array
    elif mode == "dicomsdl" : 
        dicom = dicomsdl.open(fname)
        data = dicom.pixelData()
    
    # convert range [0,1]
    data = (data-data.min())/(data.max()-data.min())
    data = (data*255).astype(np.uint8)
    data = cv2.resize(data, PNG_SIZE)

    return data
