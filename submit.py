import os
import pandas as pd
import numpy as np
import cv2
import pathlib 
import torch
from torch.utils.data import DataLoader
import argparse
from joblib import Parallel, delayed
import subprocess

# local
from rsna.utility import load_data, data_to_device, dicom2png
from rsna.preprocess import Transform, df_preprocess
from rsna.model import ResNet50Network, EfficientNet
from rsna.config import DEVICE, PLATFORM
from rsna.dataset import RSNADatasetPNG

if __name__ == "__main__" : 

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-s", "--score", type=str, default="max")
    parser.add_argument("-t", "--threshold", type=float, default=None)
    args = parser.parse_args()
    
    # DICOM --> PNG 
    try:
        os.mkdir("tmp")
    except:
        pass

    if PLATFORM == "kaggle" : 
        test_path = "/kaggle/input/rsna-breast-cancer-detection/test_images"
    elif PLATFORM == "local" : 
        test_path = "/Users/ktakeda/workspace/kaggle/rsna-breast-cancer-detection/input/rsna-breast-cancer-detection/test_images"


    def process(fname):
        img = dicom2png(str(fname), PNG_SIZE=(512,512))
        cv2.imwrite(f"tmp/{patient_id.name}/{fname.name}".replace("dcm", "png"), img)

    for patient_id in pathlib.Path(test_path).glob("*") : 
        if patient_id.name in [".DS_Store",] : continue # macOS
        try:
            os.mkdir(f"tmp/{patient_id.name}")
        except:
            pass
        # prepare file name list for Parallel
        fname_list = []
        for fname in pathlib.Path(patient_id).glob("*") : 
            if fname.name in [".DS_Store",] : continue # macOS
            fname_list.append(fname)
        _ = Parallel(n_jobs=4)(delayed(process)(fname) for fname in fname_list)
    
    # input path (png) 
    # any platform will have 'tmp' directory under the current dir
    df_test = load_data("test", custom_path="tmp")
    df_test = df_preprocess(df_test, is_train=False)
    
    # load trained model
    model = EfficientNet(pretrained=False).to(DEVICE) 
    model.load_state_dict(torch.load(f"{args.model}", map_location=torch.device(DEVICE)))
    model.eval()

    # dataset, dataloader
    transform = Transform(cfg=None, only_test=True) 
    test_dataset = RSNADatasetPNG(df_test, transform.get(is_train=False), csv_columns = ["laterality_LE", "view_LE", "age", "implant"], has_target=False, image_prep_ver="v3")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # predict
    predict_pID = {}
    for data in test_loader:
        image, meta, prediction_ids = data_to_device(data, is_train=False)

        out = model(image, meta)
        preds = torch.sigmoid(out).squeeze(1).cpu().detach().numpy()

        for prediction_id, pred in zip(prediction_ids, preds) : 
            if not predict_pID.get(prediction_id, False) :
                predict_pID[prediction_id] = [pred,]
            else :
                predict_pID[prediction_id].append(pred)

    list_prediction_id, list_target = [], []
    for k, v in predict_pID.items():
        list_prediction_id.append(k)
        list_target.append(np.max(v))

    df_submit = pd.DataFrame()
    df_submit["prediction_id"] = list_prediction_id # add new column
    if args.threshold : 
        list_target = np.array(list_target)
        list_target = np.where(list_target>args.threhold, 1, 0)
        df_submit["cancer"] = list_target 
    else:
        df_submit["cancer"] = list_target 
    df_submit = df_submit.sort_index()
    df_submit.to_csv('submission.csv', index=False)

    print(df_submit)