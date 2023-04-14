import numpy as np
import pandas as pd
import tqdm, joblib, itertools, tqdm.contrib.itertools, warnings
from tqdm.contrib import itertools

import two_class_model_evaluation
import multi_class_model_evaluation
import regression_model_evaluation
from ML_Model_Training import model_training_and_hyperparameter_tuning
from FT_D_Pipeline import ML_Pipeline
from PermutationImportance import permutation_importance

tqdm.tqdm.pandas()
warnings.filterwarnings("ignore")

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
        ml_methods,
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
        self.ml_methods = ml_methods
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

    def fit(self, permutationImportanceMethod):
        """
        permutationImportanceMethod: string
            - originalData：用原始資料放入 permutation importance
            - trainData：用訓練資料放入 permutation importance
        """
        ml = ML_Pipeline(
            ml_methods=self.ml_methods,
            inputFeatures=self.inputFeatures,
            target=self.target,
        )
        ml.fit_Pipeline(fit_data=self.trainData)
        trainData, valiData, testData = [
            ml.transform_Pipeline(transform_data=j, mode=i)
            for i, j in zip(
                ["train", "vali", "test"],
                [self.trainData, self.valiData, self.testData],
            )
        ]
        inputFeatures = trainData.columns.tolist().copy()
        inputFeatures = [i for i in inputFeatures if self.target != i]

        self.modelFit(
            trainData=trainData, valiData=valiData, inputFeatures=inputFeatures
        )
        self.evaluationResult = [
            joblib.delayed(self.modelEvaluation)(oneSet, modelName)
            for oneSet, modelName in itertools.product(
                [trainData, valiData, testData], self.modelNameList
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
        if permutationImportanceMethod is not None:
            original_PI_result, train_PI_result = None, None
            if "originalData" in permutationImportanceMethod:
                original_PI_result = [
                    permutation_importance(
                        disturbData=self.dataDict[oneSet],
                        target="target",
                        targetType="classification",
                        originalResult=Result,
                        metric="Accuracy",
                        model=f"result/{self.modelFileName}-{modelName}.gzip",
                        mlFlow=ml, 
                        disturbFeature = self.inputFeatures
                    ).fit()
                    for (oneSet, modelName), Result in zip(
                        tqdm.contrib.itertools.product(
                            ["train", "vali", "test"], self.modelNameList
                        ),
                        self.evaluationResult,
                    )
                ]
                inputFeaturesLength = original_PI_result.__len__() // (3 * self.modelNameList.__len__())
                set_model_list = [(i, j) for i, j in itertools.product(["train", "vali", "test"], self.modelNameList) for n in range(inputFeaturesLength)]
                original_PI_result = [{"Model": j[1], "Set": j[0], **i} for oneResult in original_PI_result for i, j in zip(oneResult, set_model_list)]
            if "trainData" in permutationImportanceMethod:
                dataDict = {"train": trainData, "vali": valiData, "test": testData}
                train_PI_result = [
                    permutation_importance(
                        disturbData=dataDict[oneSet],
                        target="target",
                        targetType="classification",
                        originalResult=Result,
                        metric="Accuracy",
                        model=f"result/{self.modelFileName}-{modelName}.gzip",
                    ).fit()
                    for (oneSet, modelName), Result in zip(
                        tqdm.contrib.itertools.product(
                            ["train", "vali", "test"], self.modelNameList
                        ),
                        self.evaluationResult,
                    )
                ]
                inputFeaturesLength = original_PI_result.__len__() // (3 * self.modelNameList.__len__())
                set_model_list = [(i, j) for i, j in itertools.product(["train", "vali", "test"], self.modelNameList) for n in range(inputFeaturesLength)]
                train_PI_result = [{"Model": j[1], "Set": j[0], **i} for oneResult in train_PI_result for i, j in zip(oneResult, set_model_list)]
            outputResult = {
                "Evaluation": self.evaluationResult,
                "PermutationImportance": {
                    oneMethod: oneResult
                    for oneMethod, oneResult in zip(["originalData", "trainData"], [original_PI_result, train_PI_result]) if oneMethod in permutationImportanceMethod
                }
            }
            
            return outputResult
        else:
            return {
                "Evaluation": self.evaluationResult
            }

    def modelFit(self, trainData, valiData, inputFeatures):
        # totalModelResult = [self.oneModelTraining(modelName) for modelName in self.modelNameList]
        delayedFunc = [
            joblib.delayed(self.oneModelTraining)(
                **{
                    "modelName": modelName,
                    "trainData": trainData,
                    "valiData": valiData,
                    "inputFeatures": inputFeatures,
                }
            )
            for modelName in self.modelNameList
        ]
        totalModelResult = joblib.Parallel(n_jobs=-1)(delayedFunc)
        self.modelTrainingResult = {
            modelName: oneResult
            for modelName, oneResult in zip(self.modelNameList, totalModelResult)
        }
        return

    def oneModelTraining(self, modelName, trainData, valiData, inputFeatures):
        modelTrainingObj = model_training_and_hyperparameter_tuning(
            trainData=trainData,
            valiData=valiData,
            inputFeatures=inputFeatures,
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
