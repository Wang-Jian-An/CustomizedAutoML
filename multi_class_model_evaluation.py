import numpy as np
from sklearn.metrics import *
import two_class_model_evaluation

def model_evaluation(ytrue: np.array, ypred: np.array, ypred_proba: np.array):
    
    # 把各類別分別變成二分類
    uniqueTarget = np.sort(np.unique(ytrue))
    eachClassBinaryTrue = {
        oneClass: np.where(ytrue == oneClass, 1, 0) for oneClass in uniqueTarget
    }
    eachClassBinaryPred = {
        oneClass: np.where(ypred == oneClass, 1, 0) for oneClass in uniqueTarget
    }
    eachClassResult = {
        oneClass: two_class_model_evaluation.model_evaluation(
            ytrue = eachClassBinaryTrue[oneClass],
            ypred = eachClassBinaryPred[oneClass],
            ypred_proba = ypred_proba[:, oneClassIndex]
        ) for oneClassIndex, oneClass in enumerate(uniqueTarget)
    }
    eachClassResult = {
        f"{oneClass}_{oneMeasureName}": oneMeasure \
            for oneClass, oneResult in eachClassResult.items() for oneMeasureName, oneMeasure in oneResult.items()
    }
    return eachClassResult