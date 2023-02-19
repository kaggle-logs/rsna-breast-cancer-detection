import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score

def rsna_accuracy(y_true : list, y_pred : list, threshold=0.5):
    if type(y_pred) == list : 
        y_pred = np.array(y_pred)
    y_pred = np.where(y_pred>threshold, 1, 0)
    return accuracy_score(y_true, y_pred)

def rsna_roc(y_true : list, y_pred : list):
    # if y_true has only on class, the roc_auc_score fails.
    # please be careful.
    return roc_auc_score(y_true, y_pred)

def rsna_precision_recall_f1(y_true, y_pred):
    y_pred = np.where(np.array(y_pred)>0.5,1,0)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1


def pfbeta(labels, predictions, beta=1.):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        # [0, 1] の間に値を収めるための処理
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction # True positive
        else:
            cfp += prediction # False positive

    # sanity check
    if y_true_count == 0 : 
        # print(f"WARNING ::: There is no positive events. {labels}")
        return 0

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0

def optimal_f1(labels, predictions):
    thres = np.linspace(0, 1, 101)
    f1s = [pfbeta(labels, predictions > thr) for thr in thres]
    idx = np.argmax(f1s)
    return f1s[idx], thres[idx]