import os
import numpy as np
import gzip
import joblib
import pickle
import pandas as pd
from scipy.stats import ttest_1samp
from .two_class_model_evaluation import model_evaluation as two_class_model_evaluation
from .multi_class_model_evaluation import model_evaluation as multi_class_model_evaluation
import tqdm
from .Model_Prediction import modelPrediction

class permutation_importance:
    def __init__(
        self,
        disturbData: pd.DataFrame,
        target: str,
        targetType: str,
        originalResult,
        model: list,
        metric,
        mlFlow = None, 
        disturbFeature="All",
        n_repeats=5,
    ):
        self.disturbData = disturbData.copy()
        self.target = target
        self.targetType = targetType
        self.originalResult = originalResult
        if type(model) == str:
            with gzip.GzipFile(model) as f:
                self.model = pickle.load(f)
        else:
            self.model = model
        if type(mlFlow) == str:
            pass
        else:
            self.mlFlow = mlFlow
        
        self.inputFeatures = [i for i in disturbData.columns.tolist() if self.target not in i] # 原始資料中有的特徵
        try:
            self.modelInputFeatures = [i.feature_name_ for i in self.model]
        except:
            try:
                self.modelInputFeatures = [i.feature_names_.tolist() for i in self.model]
            except:
                self.modelInputFeatures = [i.feature_names_in_.tolist() for i in self.model] # 放入模型時的特徵
        
        self.originalTarget = self.disturbData[self.target].tolist()
        self.disturbFeature = self.inputFeatures.copy() if disturbFeature == "All" else disturbFeature # 想要打亂的特徵
        self.n_repeats = n_repeats
        self.metric = "F1-Score_for_1" if metric == "f1" else metric
        return

    def fit(self):
        result = [
            {
                "Feature": oneFeature,
                "DisturbYhat": [
                    self.predict_disturb_data(oneFeature=oneFeature)
                    for i in range(self.n_repeats)
                ],
            }
            for oneFeature in self.disturbFeature
        ]
        if (
            self.targetType == "classification"
            and np.unique(self.originalTarget).tolist().__len__() == 2
        ): # 如果是二分類
            result = [
                {
                    "Feature": oneResult["Feature"],
                    "Importance_for_Each_Data": [
                        two_class_model_evaluation(
                            ytrue=self.originalTarget,
                            ypred=i["yhat"],
                            ypred_proba=i["yhat_proba"],
                        )[self.metric]
                        for i in oneResult["DisturbYhat"]
                    ],
                }
                for oneResult in result
            ]
            result = [
                {
                    "Feature": oneResult["Feature"],
                    "Mean-Importance": np.mean(
                        np.array(oneResult["Importance_for_Each_Data"])
                        - self.originalResult[self.metric]
                    ),
                    "Std-Importance": np.std(
                        np.array(oneResult["Importance_for_Each_Data"])
                        - self.originalResult[self.metric]
                    ),
                    "p-value for t-test": ttest_1samp(
                        a=np.array(oneResult["Importance_for_Each_Data"])
                        - self.originalResult[self.metric],
                        popmean=0,
                    )[1],
                    "originalMetric": round(self.originalResult[self.metric], 4),
                    "Metric_for_Each_Data": [round(i, 4) for i in oneResult["Importance_for_Each_Data"]],
                    "Importance_for_Each_Data": [round(i) for i in (np.array(
                        oneResult["Importance_for_Each_Data"]
                    )
                    - self.originalResult[self.metric])],
                }
                for oneResult in result
            ]
        elif (
            self.targetType == "classification"
            and np.unique(self.originalTarget).tolist().__len__() > 2
        ): # 如果是多分類
            result = [
                {
                    "Feature": oneResult["Feature"],
                    "Importance_for_Each_Data": [
                        multi_class_model_evaluation(
                            ytrue=self.originalTarget,
                            ypred=i["yhat"],
                            ypred_proba=i["yhat_proba"],
                        )[self.metric]
                        for i in oneResult["DisturbYhat"]
                    ],
                }
                for oneResult in result
            ]
            result = [
                {
                    "Feature": oneResult["Feature"],
                    "Mean-Importance": np.mean(
                        np.array(oneResult["Importance_for_Each_Data"])
                        - self.originalResult[self.metric]
                    ),
                    "Std-Importance": np.std(
                        np.array(oneResult["Importance_for_Each_Data"])
                        - self.originalResult[self.metric]
                    ),
                    "p-value for t-test": ttest_1samp(
                        a=np.array(oneResult["Importance_for_Each_Data"])
                        - self.originalResult[self.metric],
                        popmean=0,
                    )[1],
                    "originalMetric": round(self.originalResult[self.metric], 4),
                    "Metric_for_Each_Data": [round(i, 4) for i in oneResult["Importance_for_Each_Data"]],
                    "Importance_for_Each_Data": [round(i, 4) for i in (np.array(
                        oneResult["Importance_for_Each_Data"]
                    )
                    - self.originalResult[self.metric])],
                }
                for oneResult in result
            ]
        else:
            pass
        return result
    
    def permuteData(self, data, oneFeature): # 確定留下來
        permuteedData = data.copy()
        targetFeatureData = permuteedData[oneFeature].values.copy()
        np.random.shuffle(targetFeatureData)
        permuteedData[oneFeature] = targetFeatureData
        return permuteedData

    def predict_disturb_data(self, oneFeature): # 確定留下來
        permutedData = self.permuteData(data=self.disturbData, oneFeature=oneFeature)
        if self.mlFlow is not None: # 針對原始資料做解釋
            permutedData = self.mlFlow.transform_Pipeline(permutedData[self.inputFeatures], permutedData[self.target], mode = "test")
            permutedData = pd.concat(list(permutedData), axis = 1)
#         yhat = self.model.predict(permutedData[self.inputFeatures]).tolist()
        yhat = modelPrediction(
            modelList = self.model,
            predData = permutedData,
            targetType = self.targetType, 
            featureList = self.modelInputFeatures
        )
        if self.targetType == "classification" and permutedData[self.target].unique().tolist().__len__() == 2:
            yhat, yhat_proba = list(yhat.values())
            yhat_proba = yhat_proba[:, -1]
#             yhat_proba = self.model.predict_proba(permutedData[self.inputFeatures])[:, -1].tolist()
            return {"yhat": yhat, "yhat_proba": yhat_proba}
        elif self.targetType == "classification" and permutedData[self.target].unique().tolist().__len__() > 2:
            yhat, yhat_proba = list(yhat.values())
            yhat_proba = yhat_proba[:, -1]
#             yhat_proba = self.model.predict_proba(permutedData[self.inputFeatures]).tolist()
            return {"yhat": yhat, "yhat_proba": yhat_proba}            
        else:
            return {"yhat": yhat}