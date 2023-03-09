import os
import numpy as np
import gzip
import joblib
import pickle
import pandas as pd
from scipy.stats import ttest_1samp
import two_class_model_evaluation
import multi_class_model_evaluation
import tqdm


class permutation_importance:
    def __init__(
        self,
        disturbData: pd.DataFrame,
        target: str,
        targetType: str,
        originalResult,
        model_path_name,
        metric,
        disturbFeature="All",
        n_repeats=5,
    ):
        self.disturbData = disturbData.copy()
        with gzip.GzipFile(model_path_name) as f:
            self.model = pickle.load(f)
        try:
            self.inputFeatures = self.model.feature_name_
        except:
            self.inputFeatures = self.model.feature_names_in_
        self.target = target
        self.targetType = targetType
        self.originalTarget = self.disturbData[self.target].tolist()
        self.originalYhat = self.model.predict(self.disturbData[self.inputFeatures])
        self.originalResult = originalResult
        if disturbFeature == "All":
            self.disturbFeature = self.inputFeatures.copy()
        else:
            self.disturbFeature = disturbFeature
        self.n_repeats = n_repeats
        if metric == "f1":
            self.metric = "F1-Score_for_1"
        else:
            self.metric = metric
        return

    def permuteData(self, data, oneFeature):
        assert (
            oneFeature in self.inputFeatures
        ), "The feature interested in disturbition must be the input features. "

        permuteedData = data.copy()
        targetFeatureData = permuteedData[oneFeature].values.copy()
        np.random.shuffle(targetFeatureData)
        permuteedData[oneFeature] = targetFeatureData
        return permuteedData

    def predict_disturb_data(self, oneFeature):
        permuteedData = self.permuteData(data=self.disturbData, oneFeature=oneFeature)
        yhat = self.model.predict(permuteedData[self.inputFeatures]).tolist()
        if self.targetType == "classification":
            yhat_proba = self.model.predict_proba(permuteedData[self.inputFeatures])[
                :, -1
            ].tolist()
            return {"yhat": yhat, "yhat_proba": yhat_proba}
        else:
            return {"yhat": yhat}

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
        ):
            result = [
                {
                    "Feature": oneResult["Feature"],
                    "Importance_for_Each_Data": [
                        two_class_model_evaluation.model_evaluation(
                            ytrue=self.originalTarget,
                            ypred=i["yhat"],
                            ypred_proba=i["yhat"],
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
                    "originalMetric": self.originalResult[self.metric],
                    "Metric_for_Each_Data": oneResult["Importance_for_Each_Data"],
                    "Importance_for_Each_Data": np.array(
                        oneResult["Importance_for_Each_Data"]
                    )
                    - self.originalResult[self.metric],
                }
                for oneResult in result
            ]
        elif (
            self.targetType == "classification"
            and np.unique(self.originalTarget).tolist().__len__()
        ):
            pass
        else:
            pass
        return result
