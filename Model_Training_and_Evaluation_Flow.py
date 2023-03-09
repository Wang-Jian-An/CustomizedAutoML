import numpy as np
import pandas as pd

import tqdm, joblib, itertools, tqdm.contrib.itertools
from tqdm.contrib import itertools

tqdm.tqdm.pandas()

import warnings

warnings.filterwarnings("ignore")
import two_class_model_evaluation
import multi_class_model_evaluation
import regression_model_evaluation
from ML_Model_Training import model_training_and_hyperparameter_tuning
from PermutationImportance import permutation_importance

"""
本程式碼主旨：針對一組資料，進行模型訓練後，進行模型評估
輸入：訓練資料、驗證資料、測試資料、輸入特徵、目標變數
輸出：報表

"""


class modelTrainingFlow:
    def __init__(
        self,
        trainData: pd.DataFrame,
        valiData: pd.DataFrame,
        testData: pd.DataFrame,
        inputFeatures: list,
        target,
        targetType,
        mainMetric,
        featureSelection=None,
        featureImportance="PermutationImportance",
        modelFileName=None,
    ):
        self.trainData = trainData
        self.valiData = valiData
        self.testData = testData
        self.inputFeatures = inputFeatures
        self.target = target
        self.targetType = targetType
        self.mainMetric = mainMetric
        self.featureSelection = featureSelection
        self.featureImportance = featureImportance
        self.modelFileName = modelFileName
        self.modelTrainingResult = {}
        self.dataDict = {
            "train": self.trainData,
            "vali": self.valiData,
            "test": self.testData,
        }

        # 檢查每個參數填寫是否正確
        assert self.targetType in [
            "classification",
            "regression",
        ], "target_type must be classification or regression. "
        if self.targetType == "classification":
            assert self.mainMetric in [
                "accuracy",
                "f1",
                "auroc",
            ], "main_metric must be accuracy, f1 or auroc. "
        else:
            assert self.mainMetric in [
                "mse",
                "rmse",
            ], "main_metric must be mse or rmse. "

        if self.targetType == "classification":
            self.modelNameList = [
                "Random Forest with Entropy",
                "Random Forest with Gini",
                "ExtraTree with Entropy",
                "ExtraTree with Gini",
                "XGBoost",
                "LightGBM",
                "LightGBM with ExtraTrees",
            ]
        else:
            self.modelNameList = [
                "Random Forest with squared_error",
                "Random Forest with absolute_error",
                "Random Forest with friedman_mse",
                "ExtraTree with squared_error",
                "ExtraTree with absolute_error",
                "ExtraTree with friedman_mse",
                "XGBoost",
                "LightGBM",
                "LightGBM with ExtraTrees",
            ]
        return

    def fit(self, permutationImportance=False):
        self.modelFit()
        self.evaluationResult = [
            joblib.delayed(self.modelEvaluation)(oneSet, modelName)
            for oneSet, modelName in itertools.product(
                [self.trainData, self.valiData, self.testData], self.modelNameList
            )
        ]
        parallel = joblib.Parallel(n_jobs=-1)
        self.evaluationResult = parallel(self.evaluationResult)
        self.evaluationResult = [
            {
                "Model": modelName,
                "Features": self.modelTrainingResult[modelName]["Features"],
                "Set": oneSet,
                "Number_of_Data": self.dataDict[oneSet][self.target]
                .value_counts()
                .to_dict()
                if self.targetType == "classification"
                else self.dataDict[oneSet].shape[0],
                **Result,
            }
            for (oneSet, modelName), Result in zip(
                itertools.product(["train", "vali", "test"], self.modelNameList),
                self.evaluationResult,
            )
        ]
        if permutationImportance:
            PI_result = [
                permutation_importance(
                    disturbData=self.dataDict[oneSet],
                    target="target",
                    targetType="classification",
                    originalResult=Result,
                    metric="Accuracy",
                    model_path_name=f"result/{self.modelFileName}-{modelName}.gzip",
                ).fit()
                for (oneSet, modelName), Result in zip(
                    tqdm.contrib.itertools.product(
                        ["train", "vali", "test"], self.modelNameList
                    ),
                    self.evaluationResult,
                )
            ]
            PI_result = [i for oneResult in PI_result for i in oneResult]
            return self.evaluationResult, PI_result
        else:
            return self.evaluationResult

    def modelFit(self):
        # totalModelResult = [self.oneModelTraining(modelName) for modelName in self.modelNameList]
        delayedFunc = [
            joblib.delayed(self.oneModelTraining)(modelName)
            for modelName in self.modelNameList
        ]
        totalModelResult = joblib.Parallel(n_jobs=-1)(delayedFunc)
        self.modelTrainingResult = {
            modelName: oneResult
            for modelName, oneResult in zip(self.modelNameList, totalModelResult)
        }
        return

    def oneModelTraining(self, modelName):
        modelTrainingObj = model_training_and_hyperparameter_tuning(
            trainData=self.trainData,
            valiData=self.valiData,
            inputFeatures=self.inputFeatures,
            target=self.target,
            target_type=self.targetType,
            model_name=modelName,
            feature_selection_method=self.featureSelection,
            main_metric=self.mainMetric,
            model_file_name=f"result/{self.modelFileName}-{modelName}.gzip",
        )
        oneModelResult = modelTrainingObj.model_training()
        return oneModelResult

    def modelEvaluation(self, set, model_name):
        if self.targetType == "classification":
            yhat_test = self.modelTrainingResult[model_name]["Model"].predict(
                set[self.modelTrainingResult[model_name]["Features"]]
            )
            yhat_proba_test = self.modelTrainingResult[model_name][
                "Model"
            ].predict_proba(set[self.modelTrainingResult[model_name]["Features"]])
            if set[self.target].unique().tolist().__len__() == 2:
                one_model_all_score = two_class_model_evaluation.model_evaluation(
                    ytrue=set[self.target],
                    ypred=yhat_test,
                    ypred_proba=yhat_proba_test[:, 1],
                )
            else:
                one_model_all_score = multi_class_model_evaluation.model_evaluation(
                    ytrue=set[self.target], ypred=yhat_test, ypred_proba=yhat_proba_test
                )
        else:
            yhat = self.modelTrainingResult[model_name]["Model"].predict(
                set[self.modelTrainingResult[model_name]["Features"]]
            )
            one_model_all_score = regression_model_evaluation.model_evaluation(
                ytrue=set[self.target], ypred=yhat
            )
        return one_model_all_score