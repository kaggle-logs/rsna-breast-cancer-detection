from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, normalize

def load_data(f_name):
    # Add path column
    base_path = "/kaggle/input/rsna-breast-cancer-detection"
    
    # Read in Data
    data = pd.read_csv(f"{base_path}/{f_name}.csv")
    data["patient_id"] = data["patient_id"].apply(str)
    data["image_id"] = data["image_id"].apply(str)
    data["path"] = base_path + "/" + f_name + "_images/" + data["patient_id"]+"/"+data["image_id"]+".dcm"

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