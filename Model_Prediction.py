import numpy as np
import pandas as pd

def modelPrediction(
    modelList: list,
    predData: np.ndarray or pd.DataFrame,
    targetType: str, 
    modelName: str = None, 
    featureList: list = None
):
    assert featureList is not None or modelName is not None, "Either featureList or modelName must not be None. "
    
    if featureList is None:
        if "LightGBM" in modelName:
            featureList = [i.feature_name_ for i in modelList]
        else:
            featureList = [i.feature_names_in_ for i in modelList]
    if targetType == "classification":
        yhatProb_List = [
            i.predict_proba(predData[j]).tolist() for i, j in zip(modelList, featureList)
        ]
        yhatProb_List = [
            [
                oneModelResult[oneDataIndex]
                for oneModelResult in yhatProb_List
            ] 
            for oneDataIndex in range(predData.shape[0])
        ]
        yhatProb_List = np.array([np.mean(i, axis = 0) for i in yhatProb_List])
        yhatList = [np.argmax(i) for i in yhatProb_List.tolist()]
        return {
            "Yhat": yhatList,
            "YhatProba": yhatProb_List
        }
    else:
        yhatList = np.array([i.predict(predData[j]).tolist() for i, j in zip(modelList, featureList)]).T
        return {
            "Yhat": np.mean(yhatList, axis = 1)
        }