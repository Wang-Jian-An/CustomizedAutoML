import os
import gzip
import pickle
import tqdm
import datetime
import itertools
import tqdm.contrib.itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from monitor import ml_model
from .two_class_model_evaluation import model_evaluation as two_class_model_evaluation
from .multi_class_model_evaluation import model_evaluation as multi_class_model_evaluation
from .regression_model_evaluation import model_evaluation as regression_model_evaluation
from .ML_Model_Training import model_training_and_hyperparameter_tuning
from .FT_D_Pipeline import ML_Pipeline
from .Model_Prediction import modelPrediction
from .PermutationImportance import permutation_importance
from .MLEnv import *
tqdm.tqdm.pandas()

"""
本程式碼主旨：針對一組資料，進行模型訓練後，進行模型評估
輸入：訓練資料、驗證資料、測試資料、輸入特徵、目標變數
輸出：報表

"""

model_id = list()
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
        thresholdMetric = None, 
        modelNameList: list = None, 
        metaLearner: str = None, 
        hyperparameter_tuning_method: str = "default", 
        hyperparameter_tuning_epochs: int = 40, 
        metaLearner_hyperparameter_tuning_method: str = "default",
        metaLearner_hyperparameter_tuning_epochs: int = 40, 
        featureSelection: str = None,
        modelFilePath: str = None, 
        fitBestModel: bool = False,
        importanceMethod: str = "None", 
        importanceTarget: str = "trainData",
        device: str = "cpu",
        wandb_config: dict = None
    ):

        """
        trainData: pd.DataFrame
        valiData: pd.DataFrame
        testData: pd.DataFrame
        inputFeatures: list
        target: string
        target_type: string
        ml_methods: dict
            - key: method
            - value: 
        HTMetric: Option[string]
        thresholdMetric: Option[string],
        modelNameList: list
            - 一個 list 代表一組模型，可能是一個，可能是多個模型組合成的整合式學習
            - 範例：[["Random Forest with Entropy"], ["XGBoost", "LightGBM"]]
        hyperparameter_tuning_method: Option[string]
        hyperparameter_tuning_epochs
        featureSelection
        importanceMethod: Option[list]
            - None
            - permutationImportance
            - LIME
            - SHAPE
        importanceTarget
            - "trainData"
            - "originalData"
        """

        
        # 檢查每個參數填寫是否正確
        assert targetType in ["classification", "regression"], "targetType must be classification or regression. "
        assert hyperparameter_tuning_method in [None, "default", "TPESampler"], 'hyperparameter_tuning_method must be None, "default", "TPESampler" .'
        if targetType == "regression":
            assert HTMetric in ["MSE", "RMSE", "MAPE", "MAE"], 'main_metric must be "mse", "rmse", "mape" or "mae". '
        else: 
            pass
        assert ("None" in importanceMethod and importanceMethod.__len__() == 1) or ("None" not in importanceMethod), "None must not be includes when there are at least one method of importance. "
        
        # 初始化變數
        self.inputFeatures = inputFeatures
        self.target = target
        self.targetType = targetType
        self.ml_methods_dict = ml_methods
        self.ml_methods = [value for key, value in ml_methods.items() if value in totalMLMethodsList]
        self.regression_transform = ml_methods["regression_transform"] if "regression_transform" in ml_methods.keys() else "None"
        self.featureSelection = featureSelection
        self.modelFilePath = modelFilePath
        self.fitBestModel = fitBestModel
        self.modelTrainingResult = {}
        self.hyperparameter_tuning_method = hyperparameter_tuning_method
        self.hyperparameter_tuning_epochs = hyperparameter_tuning_epochs
        self.metaLearner_hyperparameter_tuning_method = metaLearner_hyperparameter_tuning_method,
        self.metaLearner_hyperparameter_tuning_epochs = metaLearner_hyperparameter_tuning_epochs
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
                modelNameList = [
                    ["Random Forest with Entropy"],
                    ["Random Forest with Gini"],
                    ["ExtraTree with Entropy"],
                    ["ExtraTree with Gini"],
                    ["XGBoost"],
                    ["LightGBM"],
                    ["LightGBM with ExtraTrees"],
                    ["CatBoost"]
                ]
            else:
                modelNameList = [
                    ["Random Forest with squared_error"],
                    ["Random Forest with absolute_error"],
                    ["Random Forest with friedman_mse"],
                    ["ExtraTree with squared_error"],
                    ["ExtraTree with absolute_error"],
                    ["ExtraTree with friedman_mse"],
                    ["XGBoost"],
                    ["LightGBM"],
                    ["LightGBM with ExtraTrees"],
                    ["CatBoost"]
                ]

        self.modelNameList = {
            "_".join(one_model_list): one_model_list
            for one_model_list in modelNameList
        }
        self.metaLearner = metaLearner
        self.importanceMethod = ["None"]
        self.device = device
        self.ml = ML_Pipeline(
            ml_methods=self.ml_methods,
            inputFeatures=self.inputFeatures,
            target=self.target,
        )

        self.wandb_config = dict()
        for one_config in ["project_name", "entity", "group_name"]:
            self.wandb_config[one_config] = None if wandb_config is None else (
                wandb_config[one_config] if one_config in wandb_config.keys() else None
            )
        return

    @ml_model.deco
    def one_model_train_and_eval(
        self,
        train_data,
        vali_data,
        test_data, 
        input_features,
        model_name,
        model_list,
        **kwargs
    ):

        self.modelTrainingResult = self.model_fit(
            train_data, 
            vali_data, 
            input_features, 
            model_name, 
            model_list
        )
        evaluation_result = [
            self.modelEvaluation(
                one_dataset,
                model_name
            )
            for one_dataset in [train_data, vali_data, test_data]
        ]
        if "project_name" in kwargs.keys():
            wandb_config = [
                {
                    "ID": "Flow_{}_{}".format(
                        model_id.__len__(), 
                        one_set
                    ), 
                    "Model": model_name,
                    "Set": one_set,
                    "Meta-Learner": self.metaLearner, 
                    "Features": input_features,
                    **self.ml_methods_dict, 
                    "Number_of_Data": one_target.value_counts().to_dict() if self.targetType == "classification" else one_target.values.shape[0],
                }
                for index, (one_set, one_target) in enumerate(
                    zip(
                        ["train", "vali", "test"], 
                        [self.trainTarget, self.valiTarget, self.testTarget]
                    )
                )
            ]
            model_id.append(True)
            return wandb_config, evaluation_result
        else:
            result = [
                {
                    "ID": "Flow_{}_{}_{}".format(
                        model_id.__len__() + index, 
                        str(datetime.datetime.today()).split(".")[0], 
                        one_set
                    ), 
                    "Model": model_name,
                    "Set": one_set,
                    "Meta-Learner": self.metaLearner, 
                    "Features": input_features,
                    **self.ml_methods_dict, 
                    "Number_of_Data": one_target.value_counts().to_dict() if self.targetType == "classification" else one_target.values.shape[0],
                    **one_eval_result,
                }
                for index, (one_set, one_target, one_eval_result) in enumerate(
                    zip(
                        ["train", "vali", "test"], 
                        [self.trainTarget, self.valiTarget, self.testTarget], 
                        evaluation_result
                    )
                ) 
            ]
            return result

    def fit(self):
        
        # Feature Engineer
        self.ml.fit_Pipeline(
            fit_data = pd.concat([self.trainInputData, self.trainTarget], axis = 1)
        )
        (trainInputData, trainTarget), (valiInputData, valiTarget), (testInputData, testTarget) = [
            self.ml.transform_Pipeline(transform_data=j[0], transform_target = j[1], mode=i)
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

        if self.regression_transform == "LN":
            trainData[self.target] = np.log(trainData[self.target])
            valiData[self.target] = np.log(valiData[self.target])        

        # Model Training and Evaluation
        final_model_training_and_eval_result = list()  
        for one_model_name, one_model_list in self.modelNameList.items():
            final_model_training_and_eval_result.extend(
                self.one_model_train_and_eval(
                    train_data = trainData,
                    vali_data = valiData,
                    test_data = testData, 
                    input_features = FE_inputFeatures,
                    model_name = one_model_name, 
                    model_list = one_model_list,
                    **self.wandb_config
                )                
            )

        if self.modelFilePath is not None:
            for one_model_name in self.modelNameList.keys():
                file_oneModelName = one_model_name.replace("Random Forest", "RF").replace("XGBoost", "XGB").replace("ExtraTree", "ET").replace("LightGBM", "LB").replace(" with ", "-")
                if self.metaLearner:
                    with gzip.GzipFile(os.path.join(self.modelFilePath, "{}-{}_metaLearner_{}.gzip".format("-".join(self.ml_methods), file_oneModelName, self.metaLearner) ), "wb") as f:
                        pickle.dump(self.modelTrainingResult[one_model_name], f)
                else:

                    with gzip.GzipFile(os.path.join(self.modelFilePath, "{}-{}.gzip".format("-".join(self.ml_methods), file_oneModelName) ), "wb") as f:
                        pickle.dump(self.modelTrainingResult[one_model_name], f)
        
        # Step4. Fit best model(暫時先不要用，還沒把 Baggings 功能加進去)
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

        # Step5. Model Explanation
        if "None" in self.importanceMethod:
            return {
                "Evaluation": final_model_training_and_eval_result
            }

        else:
            if "originalData" in importanceTarget:
                if "permutationImportance" in self.importanceMethod:
                    pass
                    
                elif "LIME" in self.importanceMethod:
                    pass
        
                elif "SHAP" in self.importanceMethod:
                    pass

            elif "trainData" in importanceTarget:
                if "permutationImportance" in importanceMethod:
                    pass
                    
                elif "LIME" in self.importanceMethod:
                    pass
        
                elif "SHAP" in self.importanceMethod:
                    pass
            
            outputResult = {
                "Evaluation": self.evaluationResult,
                "PermutationImportance": {
                    oneMethod: oneResult
                    for oneMethod, oneResult in zip(["originalData", "trainData"], [original_PI_result, train_PI_result])
                    if oneMethod in permutationImportanceMethod
                }
            }
            return outputResult

    def model_fit(
        self, 
        trainData, 
        valiData, 
        inputFeatures, 
        model_name, 
        model_list
    ):
        # 迴歸任務中目標值欲轉換
        # if self.regression_transform == "LN":
        #     trainData[self.target] = np.log(trainData[self.target])
        #     valiData[self.target] = np.log(valiData[self.target])
        
        # self.modelTrainingResult = {}
        # for modelName in self.modelNameList:
        print(model_name, "Training")

        # 如有 meta learner 且需要做超參數調整，則先從訓練資料中切出一部分驗證資料
        if self.metaLearner and self.metaLearner_hyperparameter_tuning_method != "default":
            trainData, metaLearnerVali = train_test_split(trainData, test_size = 0.2, shuffle = True)
        else:
            metaLearnerVali = None

        # 依照模型數量，創造出不同組合的資料集
        if model_list.__len__() > 1:
            kf = KFold(n_splits = model_list.__len__())
            trainDataList = [trainData.iloc[i[0], :] for i in kf.split(trainData)]
        else:
            trainDataList = [trainData.copy()]
        
        # 用迴圈方式逐一訓練模型
        trainedModelList = [
            model_training_and_hyperparameter_tuning(
                trainData=oneData,
                valiData=valiData,
                inputFeatures=inputFeatures,
                target=self.target,
                target_type=self.targetType,
                model_name=oneModel,
                feature_selection_method=self.featureSelection,
                HTMetric=self.HTMetric,
                hyperparameter_tuning_method = self.hyperparameter_tuning_method,
                hyperparameter_tuning_epochs = self.hyperparameter_tuning_epochs, 
                thresholdMetric = self.thresholdMetric,
                device = self.device
            ).model_training() for oneData, oneModel in zip(trainDataList, model_list)
        ] 

        # 如有 meta learner，先將原始資料放入 base learner 預測後，作為 meta data ，放入 meta learner 進行二次預測。
        if self.metaLearner:
            metaLearnerTrain = modelPrediction(
                modelList = [i["Model"] for i in trainedModelList], 
                featureList = [i["Features"] for i in trainedModelList],
                predData = trainData, 
                targetType = self.targetType,
                return_each_prediction = True
            )
            if self.targetType == "classification":
                metaLearnerTrain = pd.DataFrame(
                    metaLearnerTrain,
                    columns = [
                        "{}_value_{}".format(i, j) 
                        for j, i in enumerate(
                            list(itertools.product(
                                model_list, 
                                trainData[self.target].unique().tolist()
                            ))
                        )
                    ]
                )
            else:
                metaLearnerTrain = pd.DataFrame(
                    metaLearnerTrain["Yhat"],
                    columns = ["{}_value_{}".format(i, j) for j, i in enumerate(model_list)]
                )
            inputFeatures = metaLearnerTrain.columns.tolist()
            metaLearnerTrain[self.target] = trainData[self.target].tolist()

            metaLearnerValiYhat = modelPrediction(
                modelList = [i["Model"] for i in trainedModelList], 
                featureList = [i["Features"] for i in trainedModelList],
                predData = metaLearnerVali, 
                targetType = self.targetType,
                return_each_prediction = True
            )

            if self.targetType == "classification":
                metaLearnerValiYhat = pd.DataFrame(
                    metaLearnerValiYhat,
                    columns = [
                        "{}_value_{}".format(i, j) 
                        for j, i in enumerate(
                            list(itertools.product(
                                model_list, 
                                trainData[self.target].unique().tolist()
                            ))
                        )
                    ]
                )
            else:
                metaLearnerValiYhat = pd.DataFrame(
                    metaLearnerValiYhat["Yhat"],
                    columns = ["{}_value_{}".format(i, j) for j, i in enumerate(model_list)]
                )
            metaLearnerValiYhat[self.target] = metaLearnerVali[self.target].tolist()

            metaLearnerModel = model_training_and_hyperparameter_tuning(
                trainData=metaLearnerTrain,
                valiData=metaLearnerValiYhat,
                inputFeatures=inputFeatures,
                target=self.target,
                target_type=self.targetType,
                model_name=self.metaLearner,
                feature_selection_method=self.featureSelection,
                HTMetric=self.HTMetric,
                hyperparameter_tuning_method = self.metaLearner_hyperparameter_tuning_method,
                hyperparameter_tuning_epochs = self.metaLearner_hyperparameter_tuning_epochs, 
                thresholdMetric = self.thresholdMetric,
                device = self.device
            ).model_training()

        oneModelFeatures = [i["Features"] for i in trainedModelList]
        oneModelModel = [i["Model"] for i in trainedModelList]
        oneModelHT = [i["Hyperparameter_Tuning"] for i in trainedModelList]
        oneModelParamI = [i["Param_Importance"] for i in trainedModelList]
        oneModelBestThres = [i["Best_Thres"] for i in trainedModelList]
        oneMetaLearner = metaLearnerModel["Model"] if self.metaLearner else None
        oneMetaLearnerFeatures = metaLearnerModel["Features"] if self.metaLearner else None
        model_training_result = {
            model_name: {
                **{
                    i: j
                    for i, j in zip(
                        ["Features", "Model", "ModelNames", "Hyperparameter_Tuning", "Params_Importance", "Best_Thres"], 
                        [oneModelFeatures, oneModelModel, self.modelNameList ,oneModelHT, oneModelParamI, oneModelBestThres]
                    )
                },
                **{
                    "FeatureEngineering": self.ml,
                    "MetaLearner": oneMetaLearner,
                    "MetaLearnerFeatures": oneMetaLearnerFeatures
                }
            }
        }
        return model_training_result

    def oneModelTraining(
            self, 
            modelName, 
            trainData, 
            valiData, 
            inputFeatures
    ):
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
            hyperparameter_tuning_epochs = self.hyperparameter_tuning_epochs, 
            thresholdMetric = self.thresholdMetric,
            device = self.device
        )
        oneModelResult = modelTrainingObj.model_training()
        return oneModelResult

    def modelEvaluation(self, evaluateData, model_name):
        if self.targetType == "classification" and self.trainTarget.unique().shape[0] == 2:
            binary_class_thres = np.mean(self.modelTrainingResult[model_name]["Best_Thres"])
        else:
            binary_class_thres = None
        yhat_test = modelPrediction(
            modelList = self.modelTrainingResult[model_name]["Model"], 
            featureList = self.modelTrainingResult[model_name]["Features"],
            predData = evaluateData, 
            targetType = self.targetType,
            metaLearnerModel = self.modelTrainingResult[model_name]["MetaLearner"], 
            binary_class_thres = binary_class_thres
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