from .__init__ import *
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
    mae = np.mean(np.abs(ytrue - ypred))
    stdae = np.std((np.abs(ytrue - ypred)))
    mse = np.mean((ytrue - ypred) ** 2)
    stdse = np.std((ytrue - ypred) ** 2)
    rmse = mse ** (1/2)
    rstdse = stdse ** (1/2)
    mape = np.abs((ytrue - ypred) / (ytrue + 1e-8)).mean()
    stdape = np.abs((ytrue - ypred) / (ytrue + 1e-8)).std()
    return {
        "MAE": mae,
        "STDAE": stdae,
        "MSE": mse,
        "STDSE": stdse,
        "RMSE": rmse,
        "RSTDSE": rstdse,
        "MAPE": mape, 
        "STDAPE": stdape,
        "R2": r2
    }
