import numpy as np
import pandas as pd
from sklearn.metrics import *

def model_evaluation(ytrue: np.array, ypred: np.array, ypred_proba: np.array):

    """
    ytrue: an array with one dimension
    ypred: an array with one dimension
    ypred_proba: an array with two dimension
    """
    
    # F1-Score
    f1_score_1 = f1_score(y_true=ytrue, y_pred=ypred, pos_label=1)
    f1_score_0 = f1_score(y_true=ytrue, y_pred=ypred, pos_label=0)
    macro_f1_score = f1_score(y_true=ytrue, y_pred=ypred, average="macro")
    micro_f1_score = f1_score(y_true=ytrue, y_pred=ypred, average="micro")

    # PRC-AUC
    prc_precision_1, prc_recall_1, prc_threshold_1 = precision_recall_curve(
        y_true=ytrue, probas_pred=ypred_proba[:, -1].flatten(), pos_label=1
    )
    prc_precision_0, prc_recall_0, prc_threshold_0 = precision_recall_curve(
        y_true=ytrue, probas_pred=ypred_proba[:, -1].flatten(), pos_label=0
    )

    prc_auc_1 = auc(prc_recall_1, prc_precision_1)
    prc_auc_0 = auc(prc_recall_0, prc_precision_0)

    # Precision
    precision_0 = precision_score(y_true=ytrue, y_pred=ypred, pos_label=0)
    precision_1 = precision_score(y_true=ytrue, y_pred=ypred, pos_label=1)
    macro_precision = precision_score(y_true=ytrue, y_pred=ypred, average="macro")
    micro_precision = precision_score(y_true=ytrue, y_pred=ypred, average="micro")

    # Recall
    recall_0 = recall_score(y_true=ytrue, y_pred=ypred, pos_label=0)
    recall_1 = recall_score(y_true=ytrue, y_pred=ypred, pos_label=1)
    macro_recall = recall_score(y_true=ytrue, y_pred=ypred, average="macro")
    micro_recall = recall_score(y_true=ytrue, y_pred=ypred, average="micro")

    #  Accuracy
    accuracy = accuracy_score(y_true=ytrue, y_pred=ypred)

    # ROC-AUC
    fpr, tpr, roc_threshold = roc_curve(y_true=ytrue, y_score=ypred_proba[:, -1].flatten())
    roc_auc = roc_auc_score(y_true=ytrue, y_score=ypred_proba[:, -1].flatten())

    # Cross Entropy
    logloss = log_loss(
        y_true = ytrue,
        y_pred = ypred_proba
    )

    # Combine all
    all_score = {
        "f1_1": f1_score_1,
        "f1_0": f1_score_0,
        "f1_macro": macro_f1_score,
        "f1_micro": micro_f1_score,
        "prc_auc_1": prc_auc_1,
        "prc_auc_0": prc_auc_0,
        "precision_1": precision_1,
        "precision_0": precision_0,
        "macro_precision": macro_precision,
        "micro_precision": micro_precision,
        "recall_1": recall_1,
        "recall_0": recall_0,
        "macro_recall": macro_recall,
        "micro_recall": micro_recall,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "cross_entropy": logloss, 
        "fpr": fpr.tolist() if type(fpr) == np.ndarray else fpr,
        "tpr": tpr.tolist() if type(tpr) == np.ndarray else tpr,
        "True_value": ytrue.tolist() if type(ytrue) == np.ndarray or type(ytrue) == pd.Series else ytrue,
        "Predict_value": ypred.tolist() if type(ypred) == np.ndarray else ypred,
        "Predict_prob_value": ypred_proba[:, -1].tolist() if type(ypred_proba) == np.ndarray else ypred_proba[:, -1],
    }
    return all_score
