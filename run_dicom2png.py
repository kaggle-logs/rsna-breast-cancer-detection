
import pathlib
import tqdm
import cv2
import argparse

from rsna.utility import dicom2png

def run(size:int):
    output_path = "/kaggle/working/rsna_dicom2png/"
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
            img = dicom2png(dicom, PNG_SIZE=(size, size))
            cv2.imwrite(f"{output_path}/{patient_id}/{dicom.name}".replace("dcm", "png"), img)

if __name__ == "__main__" : 

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size", type=int, required=True)
    args = parser.parse_args()
    run(args.size)
