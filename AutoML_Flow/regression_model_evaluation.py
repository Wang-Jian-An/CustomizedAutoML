import numpy as np
from sklearn.metrics import *

def model_evaluation(ytrue: np.ndarray, ypred: np.ndarray):

    """
    輸入
        ytrue: 真實數值
        ypred: 預測數值
    輸出：
    功能：
    """

    assert np.ndim(ytrue) == 1 and np.ndim(ypred) == 1, "The dimension of ytrue and ypred must be 1. "
    assert ytrue.shape[0] == ypred.shape[0], "The length of ytrue and ypred are not the same. "

    # R^2 Score
    r2 = r2_score(y_true = ytrue, y_pred = ypred)
    mse = mean_squared_error(y_true = ytrue, y_pred = ypred)
    rmse = mean_squared_error(y_true = ytrue, y_pred = ypred, squared = False)
    mae = mean_absolute_error(y_true = ytrue, y_pred = ypred)
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }