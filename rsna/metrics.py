import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

def rsna_accuracy(y_true : list, y_pred : list, threshold=0.5):
    if type(y_pred) == list : 
        y_pred = np.array(y_pred)
    y_pred = np.where(y_pred>threshold, 1, 0)
    return accuracy_score(y_true, y_pred)

def rsna_roc(y_true : list, y_pred : list):
    # if y_true has only on class, the roc_auc_score fails.
    # please be careful.
    return roc_auc_score(y_true, y_pred)