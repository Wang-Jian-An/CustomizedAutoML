import numpy as np
import pandas as pd

from sklearn.metrics import *


def model_evaluation(ytrue: np.array, ypred: np.array, ypred_proba: np.array):
    # F1-Score
    f1_score_1 = f1_score(y_true=ytrue, y_pred=ypred, pos_label=1)
    f1_score_0 = f1_score(y_true=ytrue, y_pred=ypred, pos_label=0)
    macro_f1_score = f1_score(y_true=ytrue, y_pred=ypred, average="macro")
    micro_f1_score = f1_score(y_true=ytrue, y_pred=ypred, average="micro")

    # PRC-AUC
    prc_precision_1, prc_recall_1, prc_threshold_1 = precision_recall_curve(
        y_true=ytrue, probas_pred=ypred_proba, pos_label=1
    )
    prc_precision_0, prc_recall_0, prc_threshold_0 = precision_recall_curve(
        y_true=ytrue, probas_pred=ypred_proba, pos_label=0
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
    fpr, tpr, roc_threshold = roc_curve(y_true=ytrue, y_score=ypred_proba)
    roc_auc = roc_auc_score(y_true=ytrue, y_score=ypred_proba)

    # Combine all
    all_score = {
        "F1-Score_for_1": f1_score_1,
        "F1-Score_for_0": f1_score_0,
        "Macro F1-Score": macro_f1_score,
        "Micro F1-Score": micro_f1_score,
        "prc_auc_1": prc_auc_1,
        "prc_auc_0": prc_auc_0,
        "Precision_for_1": precision_1,
        "Precision_for_0": precision_0,
        "Macro Precision": macro_precision,
        "Micro Precision": micro_precision,
        "Recall_for_1": recall_1,
        "Recall_for_0": recall_0,
        "Macro Recall": macro_recall,
        "Micro Recall": micro_recall,
        "Accuracy": accuracy,
        "ROC-AUC": roc_auc,
        "fpr": fpr.tolist() if type(fpr) == np.ndarray else fpr,
        "tpr": tpr.tolist() if type(tpr) == np.ndarray else tpr,
        "True_value": ytrue.tolist() if type(ytrue) == np.ndarray else ytrue,
        "Predict_value": ypred.tolist() if type(ypred) == np.ndarray else ypred,
        "Predict_prob_value": ypred_proba.tolist() if type(ypred_proba) == np.ndarray else ypred_proba,
    }
    return all_score
