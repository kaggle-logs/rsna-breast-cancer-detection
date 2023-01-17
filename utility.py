import pandas as pd

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, normalize

from config import INPUT_PATH, DEVICE

def load_data(f_name):
    # Add path column
    
    # Read in Data
    data = pd.read_csv(f"{INPUT_PATH}/{f_name}.csv")
    data["patient_id"] = data["patient_id"].apply(str)
    data["image_id"] = data["image_id"].apply(str)
    data["path"] = INPUT_PATH + "/" + f_name + "_images/" + data["patient_id"] + "/" + data["image_id"] +".dcm"
    data["path"] = "input/rsna-breast-cancer-detection/train_images/10006/462822612.dcm"

    return data
    
    
def preprocess(data, is_train):

    if is_train : 
        # Keep only columns in test + target variable
        data = data[["patient_id", "image_id", "laterality", "view", "age", "implant", "path", "cancer"]]
    else :
        data = data[["patient_id", "image_id", "laterality", "view", "age", "implant", "path"]]

    # Encode categorical variables
    le_laterality = LabelEncoder()
    le_view = LabelEncoder()

    data['laterality'] = le_laterality.fit_transform(data['laterality'])
    data['view'] = le_view.fit_transform(data['view'])

    # print("Number of missing values in Age:", data["age"].isna().sum())
    data['age'] = data['age'].fillna(int(data["age"].mean()))
    
    return data


def data_to_device(data):
    image, metadata, targets = data.values()
    return image.to(DEVICE), metadata.to(DEVICE), targets.to(DEVICE)
