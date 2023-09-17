import os, gzip, pickle
import numpy as np
import pandas as pd
import tqdm, joblib, itertools, tqdm.contrib.itertools, warnings
from tqdm.contrib import itertools

from .two_class_model_evaluation import model_evaluation as two_class_model_evaluation
from .multi_class_model_evaluation import model_evaluation as multi_class_model_evaluation
from .regression_model_evaluation import model_evaluation as regression_model_evaluation
from .ML_Model_Training import model_training_and_hyperparameter_tuning
from .FT_D_Pipeline import ML_Pipeline
from .Model_Prediction import modelPrediction
from .PermutationImportance import permutation_importance

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
        HTMetric,
        thresholdMetric, 
        modelNameList: list = None, 
        num_baggings: int = 1, 
        hyperparameter_tuning_method = None, 
        featureSelection=None,
        modelFilePath = None, 
        fitBestModel = False, 
        regression_transform: str = None
    ):
        
        # 檢查每個參數填寫是否正確
        assert targetType in ["classification", "regression"], "target_type must be classification or regression. "
        assert hyperparameter_tuning_method in [None, "default", "TPESampler"], 'hyperparameter_tuning_method must be None, "default", "TPESampler" .'
        if targetType == "classification":
            # assert HTMetric in ["accuracy", "f1", "roc_auc", "auprc"], 'main_metric must be "accuracy", "f1", "roc_auc", "auprc". '
            assert regression_transform is None, "regression_transform must be None when target_type is classification. "
        elif targetType == "regression":
            assert HTMetric in ["mse", "rmse"], 'main_metric must be "mse", "rmse". '
        
        # 初始化變數
        self.inputFeatures = inputFeatures
        self.target = target
        self.targetType = targetType
        self.ml_methods = ml_methods
        self.num_baggings = num_baggings
        self.featureSelection = featureSelection
        self.modelFilePath = modelFilePath
        self.fitBestModel = fitBestModel
        self.modelTrainingResult = {}
        self.regression_transform = regression_transform
        if hyperparameter_tuning_method is None:
            self.hyperparameter_tuning_method = "default"
        else:
            self.hyperparameter_tuning_method = hyperparameter_tuning_method
        self.trainInputData = trainData[self.inputFeatures].reset_index(drop = True)
        self.valiInputData = valiData[self.inputFeatures].reset_index(drop = True)
        self.testInputData = testData[self.inputFeatures].reset_index(drop = True)
        self.trainTarget = trainData[self.target].reset_index(drop = True)
        self.valiTarget = valiData[self.target].reset_index(drop = True)
        self.testTarget = testData[self.target].reset_index(drop = True)
        self.HTMetric = HTMetric
        self.thresholdMetric = thresholdMetric

        if modelNameList is None:
            if self.targetType == "classification":
                self.modelNameList = [
                    "Random Forest with Entropy",
                    "Random Forest with Gini",
                    "ExtraTree with Entropy",
                    "ExtraTree with Gini",
                    "XGBoost",
                    "LightGBM",
                    "LightGBM with ExtraTrees",
                    "CatBoost"
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
                    "CatBoost"
                ]
        else:
            self.modelNameList = modelNameList
        return

    def fit(
        self, 
        permutationImportanceMethod: str = None
    ):
        """
        permutationImportanceMethod: string
            - originalData：用原始資料放入 permutation importance
            - trainData：用訓練資料放入 permutation importance
        """

        # Step1. Feature Engineer
        ml = ML_Pipeline(
            ml_methods=self.ml_methods,
            inputFeatures=self.inputFeatures,
            target=self.target,
        )
        ml.fit_Pipeline(fit_data=self.trainInputData)
        (trainInputData, trainTarget), (valiInputData, valiTarget), (testInputData, testTarget) = [
            ml.transform_Pipeline(transform_data=j[0], transform_target = j[1], mode=i)
            for i, j in zip(
                ["train", "vali", "test"],
                [
                    (self.trainInputData, self.trainTarget),
                    (self.valiInputData, self.valiTarget),
                    (self.testInputData, self.testTarget)
                ],
            )
        ]
        trainData = pd.concat([trainInputData, trainTarget], axis = 1)
        valiData = pd.concat([valiInputData, valiTarget], axis = 1)
        testData = pd.concat([testInputData, testTarget], axis = 1)
        FE_inputFeatures = trainInputData.columns.tolist().copy()# 更新模型輸入特徵，以適應特徵工程產生或移除的特徵。
        FE_inputFeatures = [i for i in FE_inputFeatures if self.target != i]

        # Step2. Model Training
        self.modelFit(
            trainData = trainData, 
            valiData = valiData, 
            inputFeatures=FE_inputFeatures
        )   
        self.modelNameList = list(self.modelTrainingResult.keys())
        
        self.evaluationResult = [
            self.modelEvaluation(oneSet, modelName)
            for oneSet, modelName in itertools.product(
                [trainData, valiData, testData], self.modelNameList
            )
        ]
        self.evaluationResult = [
            {
                "Model": modelName,
                "Features": FE_inputFeatures,
                "Set": oneSet,
                "Number_of_Data": oneTarget.value_counts().to_dict() if self.targetType == "classification" else oneTarget.values.shape[0],
                "Num_Baggings": self.num_baggings, 
                **Result,
            }
            for ((oneSet, oneTarget), modelName), Result in zip(
                itertools.product(
                    zip(["train", "vali", "test"], [self.trainTarget, self.valiTarget, self.testTarget]), 
                    self.modelNameList
                ),
                self.evaluationResult,
            )
        ]

        if self.modelFilePath is not None:
            for oneModelName in self.modelNameList:
                with gzip.GzipFile(os.path.join(self.modelFilePath, "{}-{}.gzip".format("-".join(self.ml_methods), oneModelName) ), "wb") as f:
                    pickle.dump(self.modelTrainingResult[oneModelName]["Model"], f)
        
        # Step3. Fit best model(暫時先不要用，還沒把 Baggings 功能加進去)
        if self.fitBestModel:
            test_evaluation = [
                oneEvaluation
                for oneEvaluation in self.evaluationResult if oneEvaluation["Set"] == "test"
            ]
            oneMetricList = [i[self.HTMetric] for i in test_evaluation]
            bestEvaluation = np.argmax(oneMetricList)
            bestModel = test_evaluation[bestEvaluation]["Model"]
            oneBestResult = self.oneModelTraining(
                modelName = bestModel, 
                trainData = pd.DataFrame( trainData.to_dict("records") + valiData.to_dict("records") ), 
                valiData = testData, 
                inputFeatures = self.inputFeatures,
            )

        # Step4. Model Explanation
        if permutationImportanceMethod is not None:
            original_PI_result, train_PI_result = None, None
            if "originalData" in permutationImportanceMethod:
                original_PI_result = [
                    permutation_importance(
                        disturbData = oneData,
                        target = self.target,
                        targetType="classification",
                        originalResult=Result,
                        metric = self.HTMetric,
                        model= self.modelTrainingResult[modelName]["Model"],
                        mlFlow=ml, 
                        disturbFeature = self.inputFeatures
                    ).fit()
                    for ((oneSet, oneData), modelName), Result in zip(
                        tqdm.contrib.itertools.product(
                            zip(
                                ["train", "vali", "test"], 
                                [
                                    pd.concat([self.trainInputData, self.trainTarget], axis = 1),
                                    pd.concat([self.valiInputData, self.valiTarget], axis = 1),
                                    pd.concat([self.testInputData, self.testTarget], axis = 1)
                                ]
                            ),
                            self.modelNameList
                        ),
                        self.evaluationResult,
                    )
                ]
                inputFeaturesLength = original_PI_result.__len__() // (3 * self.modelNameList.__len__())
                set_model_list = [
                    (i, j) for i, j in itertools.product(["train", "vali", "test"], self.modelNameList) for n in range(inputFeaturesLength)
                ]
                original_PI_result = [
                    {"Model": j[1], "Set": j[0], **i} for oneResult in original_PI_result for i, j in zip(oneResult, set_model_list)
                ]
            if "trainData" in permutationImportanceMethod:
#                 dataDict = {"train": trainData, "vali": valiData, "test": testData}
                train_PI_result = [
                    permutation_importance(
                        disturbData = oneData,
                        target = self.target,
                        targetType="classification",
                        originalResult=Result,
                        metric=self.HTMetric,
                        model= self.modelTrainingResult[modelName]["Model"],
                    ).fit()
                    for ((oneSet, oneData), modelName), Result in zip(
                        tqdm.contrib.itertools.product(
                            zip(
                                ["train", "vali", "test"],
                                [trainData, valiData, testData]
                            ), 
                            self.modelNameList
                        ),
                        self.evaluationResult,
                    )
                ]
                inputFeaturesLength = train_PI_result.__len__() // (3 * self.modelNameList.__len__())
                set_model_list = [
                    (i, j) for i, j in itertools.product(["train", "vali", "test"], self.modelNameList) for n in range(inputFeaturesLength)
                ]
                train_PI_result = [
                    {"Model": j[1], "Set": j[0], **i} for oneResult in train_PI_result for i, j in zip(oneResult, set_model_list)
                ]
            outputResult = {
                "Evaluation": self.evaluationResult,
                "PermutationImportance": {
                    oneMethod: oneResult
                    for oneMethod, oneResult in zip(["originalData", "trainData"], [original_PI_result, train_PI_result])
                    if oneMethod in permutationImportanceMethod
                }
            }
            
            return outputResult
        else:
            return {
                "Evaluation": self.evaluationResult
            }

    def modelFit(self, trainData, valiData, inputFeatures):
        if self.regression_transform == "LN":
            trainData[self.target] = np.log(trainData[self.target])
            valiData[self.target] = np.log(valiData[self.target])        
        self.modelTrainingResult = {}
        for modelName in self.modelNameList:
            keyName = modelName if self.num_baggings == 1 else f"{modelName}_baggings"
            baggings_trainData = [trainData, *[trainData.sample(frac = 0.5, random_state = i) for i in range(2, self.num_baggings+1, 1)]]
            print(keyName, "Training")
            trainedModelList = [
                self.oneModelTraining(
                    modelName = modelName,
                    trainData = oneData,    
                    valiData = valiData,
                    inputFeatures = inputFeatures
                ) for oneData in baggings_trainData
            ] 
            oneModelFeatures = [i["Features"] for i in trainedModelList]
            oneModelModel = [i["Model"] for i in trainedModelList]
            oneModelHT = [i["Hyperparameter_Tuning"] for i in trainedModelList]
            oneModelParamI = [i["Param_Importance"] for i in trainedModelList]
            oneModelBestThres = [i["Best_Thres"] for i in trainedModelList]
            self.modelTrainingResult = {
                **self.modelTrainingResult,
                keyName: {
                    i: j
                    for i, j in zip(
                        ["Features", "Model", "Hyperparameter_Tuning", "Params_Importance", "Best_Thres"], 
                        [oneModelFeatures, oneModelModel, oneModelHT, oneModelParamI, oneModelBestThres]
                    )
                }
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
            HTMetric=self.HTMetric,
            hyperparameter_tuning_method = self.hyperparameter_tuning_method,
            thresholdMetric = self.thresholdMetric
        )
        oneModelResult = modelTrainingObj.model_training()
        return oneModelResult

    def modelEvaluation(self, evaluateData, model_name):
        print(self.modelTrainingResult[model_name]["Best_Thres"])
        yhat_test = modelPrediction(
            modelName = model_name, 
            modelList = self.modelTrainingResult[model_name]["Model"],
            predData = evaluateData, 
            targetType = self.targetType,
            binary_class_thres = np.mean(self.modelTrainingResult[model_name]["Best_Thres"])
        )
        if self.targetType == "classification":
            yhat_test, yhat_proba_test = list(yhat_test.values())
            if evaluateData[self.target].unique().tolist().__len__() == 2:
                one_model_all_score = two_class_model_evaluation(
                    ytrue=evaluateData[self.target],
                    ypred=yhat_test,
                    ypred_proba=yhat_proba_test,
                )
            else:
                one_model_all_score = multi_class_model_evaluation(
                    ytrue=evaluateData[self.target], ypred=yhat_test, ypred_proba=yhat_proba_test
                )
        else:
            yhat_test = yhat_test["Yhat"]
            if self.regression_transform == "LN":
                yhat_test = np.exp(yhat_test)
            one_model_all_score = regression_model_evaluation(
                ytrue=evaluateData[self.target], ypred=yhat_test
            )
        return {
            **one_model_all_score,
            "Best_Threshold": self.modelTrainingResult[model_name]["Best_Thres"]
        }