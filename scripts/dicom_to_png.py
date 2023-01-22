import pydicom
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import os
from tqdm import tqdm

def dicom_to_png(fname, PNG_SIZE=(256,256)):
    dicom = pydicom.dcmread(fname)
    data = dicom.pixel_array
    
    # convert range [0,1]
    data = (data-data.min())/(data.max()-data.min())
    data = (data*255).astype(np.uint8)
    data = cv2.resize(data, PNG_SIZE)

    return data


def run():
    output_path = "/kaggle/working/rsna_dicom_to_png/"
    try:
        os.mkdir(output_path)
    except:
        pass

    for dir_patient in tqdm(pathlib.Path("/kaggle/input/rsna-breast-cancer-detection/train_images/").glob("*")):
        patient_id = dir_patient.name
        try:
            os.mkdir(f"{output_path}/{patient_id}")
        except:
            pass
        print(dir_patient)
        for dicom in pathlib.Path(f"{dir_patient}").glob("*.dcm"):
            img = dicom_to_png(dicom)
            cv2.imwrite(f"{output_path}/{patient_id}/{dicom.name}".replace("dcm", "png"), img)

if __name__ == "__main__" : 
    run()   
