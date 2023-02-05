import os
import pandas as pd
import cv2
import pathlib 
import torch
from torch.utils.data import DataLoader
import argparse
from joblib import Parallel, delayed

# local
from rsna.utility import load_data, preprocess, data_to_device, dicom2png
from rsna.model import ResNet50Network, EfficientNet
from rsna.config import DEVICE, PLATFORM
from rsna.dataset import RSNADatasetPNG

if __name__ == "__main__" : 

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
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
    prediction_id = df_test["patient_id"] + "_" + df_test["laterality"] 
    df_test = preprocess(df_test, is_train=False)
    
    # load trained model
    # model = ResNet50Network(output_size=1, num_columns=4, is_train=False).to(DEVICE) 
    model = EfficientNet(output_size=1, num_columns=4, pretrained=False, is_train=False).to(DEVICE) 
    model.load_state_dict(torch.load(f"{args.model}", map_location=torch.device(DEVICE)))
    model.eval()

    # dataset, dataloader
    test_dataset = RSNADatasetPNG(df_test, 0, 0, csv_columns=["laterality", "view", "age", "implant"], is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # predict
    predict_list = []
    for data in test_loader:
        image, meta = data_to_device(data, is_train=False)

        out = model(image, meta)
        pred = torch.sigmoid(out)

        predict_list.extend(pred.detach().numpy().flatten())

    df_submit = pd.DataFrame()
    df_submit["prediction_id"] = prediction_id # add new column
    df_submit["cancer"] = predict_list
    df_submit = df_submit.groupby("prediction_id").max()
    df_submit = df_submit.sort_index()
    df_submit.to_csv('submission.csv', index=True)

    print(df_submit)
