import os
import pandas as pd
import cv2
import pathlib 
import torch
from torch.utils.data import DataLoader

# local
from scripts.dicom_to_png import dicom_to_png
from utility import load_data, preprocess, data_to_device
from model import ResNet50Network
from config import DEVICE, PLATFORM
from dataset import RSNADatasetPNG

if __name__ == "__main__" : 
    
    # DICOM --> PNG 
    try:
        os.mkdir("tmp")
    except:
        pass
    for patient_id in pathlib.Path("input/rsna-breast-cancer-detection/test_images").glob("*") : 
        try:
            os.mkdir(f"tmp/{patient_id.name}")
        except:
            pass
        for fname in pathlib.Path(patient_id).glob("*") : 
            img = dicom_to_png(fname)
            cv2.imwrite(f"tmp/{patient_id.name}/{fname.name}".replace("dcm", "png"), img)
    
    # input path (png) 
    # any platform will have 'tmp' directory under the current dir
    df_test = load_data("test", custom_path="tmp")
    df_test = preprocess(df_test, is_train=False)
    
    # load trained model
    model = ResNet50Network(output_size=1, num_columns=4).to(DEVICE) 
    model.load_state_dict(torch.load("Fold1_Epoch1_ValidAcc0.981_ROC0.507.pth", map_location=torch.device(DEVICE)))
    model.eval()

    # dataset, dataloader
    test_dataset = RSNADatasetPNG(df_test, 0, 0, csv_columns=["laterality", "view", "age", "implant"], is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    # predict
    predict_list = []
    for data in test_loader:
        image, meta = data_to_device(data, is_train=False)

        out = model(image, meta)
        pred = torch.sigmoid(out)

        predict_list.append(pred.detach().numpy())

    df_submit = pd.DataFrame()
    df_submit["patient_id"] = ["10008_L", "10008_R"]
    df_submit["cancer"] = [ float((predict_list[0]+predict_list[1])/2), float((predict_list[2]+predict_list[3])/2)]

    df_submit.to_csv('submission.csv', index=False)