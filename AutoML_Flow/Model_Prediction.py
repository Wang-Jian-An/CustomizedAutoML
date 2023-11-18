import numpy as np
import pandas as pd

def modelPrediction(
    modelList: list,
    predData: np.ndarray or pd.DataFrame,
    targetType: str, 
    featureList, 
    return_each_prediction: bool = False, 
    metaLearnerModel = None, 
    binary_class_thres: float = 0.5, 
):
    
    """
    Model prediction using machine learning with baggings or stacking. 

    Parameters: 
    --------------
    modelList: list
        - 
    predData: np.ndarray or pd.DataFrame
        - 
    targetType: str
        - "classification" or "regression"
    featureList: list
        - Features using model inference. 
    return_each_prediction
        - 
    metaLearnerModel
        - 
    binary_class_thres
    



    Returns: 
    --------------

    
    """

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
        if return_each_prediction:
            yhatProb_List = np.array(yhatProb_List)
            return yhatProb_List.reshape((yhatProb_List.shape[0], -1))
        else:
            yhatProb_List = np.array([np.mean(i, axis = 0) for i in yhatProb_List])
            if yhatProb_List.shape[1] == 2: # 若是二分類目標
                yhatList = np.where(yhatProb_List[:, -1] > binary_class_thres, 1, 0).tolist()
            else: # 否則是多分類目標
                yhatList = np.argmax(yhatProb_List, axis = 1)
            return {
                "Yhat": yhatList,
                "YhatProba": yhatProb_List
            }
    else:
        yhat = np.array([i.predict(predData[j]).tolist() for i, j in zip(modelList, featureList)]).T
        if return_each_prediction or metaLearnerModel:
            if metaLearnerModel:
                yhat = metaLearnerModel.predict(yhat)
            return {
                "Yhat": yhat
            }
        else:
            return {
                "Yhat": np.mean(yhat, axis = 1)
            }