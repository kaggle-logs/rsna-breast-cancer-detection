import os
import pathlib
import tqdm
import cv2
import argparse
import joblib 

from rsna.utility import dicom2png, PLATFORM

def run(size:int):
    if PLATFORM == "kaggle" : 
        output_path = f"/kaggle/working/dicom2png_{size}/"
        search_path = "/kaggle/input/rsna-breast-cancer-detection/train_images/"
    elif PLATFORM == "local" : 
        output_path = "./tmp_dicom2png"
        search_path = "./input/rsna-breast-cancer-detection/train_images/"

    try:
        os.mkdir(output_path)
    except:
        pass

    for dir_patient in pathlib.Path(search_path).glob("*"):
        print(dir_patient)

        patient_id = dir_patient.name
        try:
            os.mkdir(f"{output_path}/{patient_id}")
        except:
            pass
            
        def for_joblib(fname):
            img = dicom2png(str(fname), PNG_SIZE=(size, size))
            cv2.imwrite(f"{output_path}/{patient_id}/{fname.name}".replace("dcm", "png"), img)

        joblib.Parallel(n_jobs=-1)(joblib.delayed(for_joblib)(dicom) for dicom in pathlib.Path(f"{str(dir_patient)}").glob("*.dcm"))

if __name__ == "__main__" : 

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size", type=int, required=True)
    args = parser.parse_args()
    run(args.size)
