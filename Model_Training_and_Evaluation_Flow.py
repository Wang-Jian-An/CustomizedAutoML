import numpy as np
import pandas as pd

import joblib
from tqdm.contrib import itertools
from tqdm import tqdm
tqdm.pandas()

from sklearn.metrics import *
from sklearn.inspection import permutation_importance 
from lime import lime_tabular
import itertools

import warnings
warnings.filterwarnings("ignore")
import two_class_model_evaluation
import multi_class_model_evaluation
import regression_model_evaluation
from ML_Model_Training import model_training_and_hyperparameter_tuning

"""
本程式碼主旨：針對一組資料，進行模型訓練後，進行模型評估
輸入：訓練資料、驗證資料、測試資料、輸入特徵、目標變數
輸出：報表

"""



class modelTrainingFlow():
    def __init__(
            self,
            trainData: pd.DataFrame, 
            valiData: pd.DataFrame, 
            testData: pd.DataFrame, 
            inputFeatures: list, 
            target, 
            targetType,
            mainMetric, 
            featureSelection = None, 
            featureImportance = "PermutationImportance",
            modelFileName = None
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
            "test": self.testData
        }

        # 檢查每個參數填寫是否正確
        assert self.targetType in ["classification", "regression"], "target_type must be classification or regression. "
        if self.targetType == "classification":
            assert self.mainMetric in ["accuracy", "f1", "auroc"], "main_metric must be accuracy, f1 or auroc. "
        else:
            assert self.mainMetric in ["mse", "rmse"], "main_metric must be mse or rmse. "

        if self.targetType == "classification":
            self.modelNameList = ["Random Forest with Entropy", "Random Forest with Gini", "ExtraTree with Entropy", "ExtraTree with Gini", "XGBoost", "LightGBM", "LightGBM with ExtraTrees"]
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
                "LightGBM with ExtraTrees"
            ]
        return

    def fit(self):
        self.modelFit()
        self.evaluationResult = [
            joblib.delayed(self.modelEvaluation)(oneSet, modelName) \
                for oneSet, modelName in itertools.product(
                    [self.trainData, self.valiData, self.testData], 
                    self.modelNameList
                )
        ]
        parallel = joblib.Parallel(n_jobs = -1)
        self.evaluationResult = parallel(self.evaluationResult)
        self.evaluationResult = [
            {
                "Model": modelName,
                "Features": self.modelTrainingResult[modelName]["Features"],
                "Set": oneSet,
                "Number_of_Data": self.dataDict[oneSet][self.target].value_counts().to_dict() if self.targetType == "classification" else self.dataDict[oneSet].shape[0],
                **Result
            } for (oneSet, modelName), Result in zip(
                itertools.product(["train", "vali", "test"], self.modelNameList), 
                self.evaluationResult
            )
        ]  
        return self.evaluationResult

    def modelFit(self):
        # totalModelResult = [self.oneModelTraining(modelName) for modelName in self.modelNameList]
        delayedFunc = [joblib.delayed(self.oneModelTraining)(modelName) for modelName in self.modelNameList]
        totalModelResult = joblib.Parallel(n_jobs = -1)(delayedFunc)
        self.modelTrainingResult = {
            modelName: oneResult for modelName, oneResult in zip(self.modelNameList, totalModelResult)
        }
        return 
    
    def oneModelTraining(self, modelName):
        modelTrainingObj = model_training_and_hyperparameter_tuning(
            trainData = self.trainData,
            valiData = self.valiData,
            inputFeatures = self.inputFeatures,
            target = self.target,
            target_type = self.targetType,
            model_name = modelName,
            feature_selection_method = self.featureSelection,
            main_metric = self.mainMetric,
            model_file_name = self.modelFileName
        )
        oneModelResult = modelTrainingObj.model_training()
        # self.modelTrainingResult = {
        #     **self.modelTrainingResult,
        #     modelName: oneModelResult
        # }  
        return oneModelResult

    def modelEvaluation(self, set, model_name):

        if self.targetType == "classification":
            yhat_test = self.modelTrainingResult[model_name]["Model"].predict(set[self.modelTrainingResult[model_name]["Features"]])
            yhat_proba_test = self.modelTrainingResult[model_name]["Model"].predict_proba(set[self.modelTrainingResult[model_name]["Features"]])
            if set[self.target].unique().tolist().__len__() == 2:
                one_model_all_score = two_class_model_evaluation.model_evaluation(
                    ytrue = set[self.target],
                    ypred = yhat_test,
                    ypred_proba = yhat_proba_test[:, 1]
                )
            else:
                one_model_all_score = multi_class_model_evaluation.model_evaluation(
                    ytrue = set[self.target],
                    ypred = yhat_test,
                    ypred_proba = yhat_proba_test
                )
        else:
            yhat = self.modelTrainingResult[model_name]["Model"].predict(set[self.modelTrainingResult[model_name]["Features"]]) 
            one_model_all_score = regression_model_evaluation.model_evaluation(ytrue = set[self.target],
                                                                               ypred = yhat)
        return one_model_all_score

# def model_fit(trainData: pd.DataFrame, 
#               valiData: pd.DataFrame, 
#               testData: pd.DataFrame, 
#               input_features, 
#               target_label, 
#               target_type,
#               main_metric, 
#               feature_selection_method = None, 
#               hyperparameter_tuning = "bayesopt", 
#               feature_importances = "PermutationImportance",
#               model_file_name = None):
    
#     # 檢查每個參數填寫是否正確
#     assert target_type in ["classification", "regression"], "target_type must be classification or regression. "
#     if target_type == "classification":
#         assert main_metric in ["accuracy", "f1", "auroc"], "main_metric must be accuracy, f1 or auroc. "
#     else:
#         assert main_metric in ["mse", "rmse"], "main_metric must be mse or rmse. "

#     totalResult = list()
#     totalFeatureImportanceResult = list()

#     # Step2. 使用 TabularPredictor 進行模型訓練
#     if target_type == "classification":
#         model_name_list = ["Random Forest with Entropy", "Random Forest with Gini", "ExtraTree with Entropy", "ExtraTree with Gini", "XGBoost", "LightGBM", "LightGBM with ExtraTrees"]
#     else:
#         model_name_list = ["Random Forest with squared_error", "Random Forest with absolute_error", "Random Forest with Friedman_mse", 
#                            "ExtraTree with squared_error", "ExtraTree with absolute_error", "ExtraTree with Friedman_mse",
#                            "XGBoost", "LightGBM", "LightGBM with ExtraTrees"][-2:]

#     # predictor = {
#     #     model_name: oneModelTraining(trainData = trainData,
#     #                                 valiData = valiData,
#     #                                 inputFeatures = input_features,
#     #                                 target = target_label,
#     #                                 target_type = target_type,
#     #                                 model_name = model_name,
#     #                                 feature_selection_method = feature_selection_method,
#     #                                 main_metric = main_metric,
#     #                                 model_file_name = model_file_name) for model_name in model_name_list
#     # }
#     predictor = [
#         joblib.delayed(oneModelTraining)(trainData = trainData,
#                                                     valiData = valiData,
#                                                     inputFeatures = input_features,
#                                                     target = target_label,
#                                                     target_type = target_type,
#                                                     model_name = model_name,
#                                                     feature_selection_method = feature_selection_method,
#                                                     main_metric = main_metric,
#                                                     model_file_name = model_file_name) for model_name in model_name_list
#     ]
#     print(predictor[0])
#     joblib.Parallel()(predictor)
#     predictor = {
#         model_name: modelObj[-1] for model_name, modelObj in zip(model_name_list, predictor)
#     }
#     print(predictor)

#     # Step4. 模型評估
#     for one_model_name, (set_name, set) in itertools.product(model_name_list, 
#                                                              zip(["train", "vali", "test"], 
#                                                                  [trainData, valiData, testData])):
#         # print(f"Get {one_model_name}, {set_name} evaluation")
#         basic_information = {
#             "Model": one_model_name,
#             "Features": predictor[one_model_name]["Features"],
#             "Set": set_name,
#             "Number_of_Data": set.shape[0]
#         }
#         # Step3. 將測試資料放入訓練好的模型作預測
#         if target_type == "classification":
#             yhat_test = predictor[one_model_name]["Model"].predict(set[predictor[one_model_name]["Features"]])
#             yhat_proba_test = predictor[one_model_name]["Model"].predict_proba(set[predictor[one_model_name]["Features"]])
#             one_model_all_score = two_class_model_evaulation.model_evaluation(ytrue = set[target_label],
#                                                                             ypred = yhat_test,
#                                                                             ypred_proba = yhat_proba_test[:, 1])
#         else:
#             yhat = predictor[one_model_name]["Model"].predict(set[predictor[one_model_name]["Features"]]) 
#             one_model_all_score = regression_model_evaluation.model_evaluation(ytrue = set[target_label],
#                                                                                ypred = yhat)
#         totalResult.append({**basic_information, **one_model_all_score})

#         # Step5. 變數重要性
#         if feature_importances == "PermutationImportance":
#             # print(f"Get {one_model_name}, {set_name} feature importances")
#             feature_importances_information = {
#                 "Model": one_model_name,
#                 "Set": set_name
#             }

#             feature_importance_result = permutation_importance(predictor[one_model_name]["Model"],
#                                                                X = set[predictor[one_model_name]["Features"]],
#                                                                y = set[target_label],
#                                                                n_jobs = -1)
#             totalFeatureImportanceResult += [{**feature_importances_information, 
#                                               **{"Feature": one_feature, "Importance_Mean": mean, "Importance_Std": std, "Importances": original}}\
#                                                   for one_feature, mean, std, original in zip(predictor[one_model_name]["Features"], *[feature_importance_result[i].tolist() for i in feature_importance_result.keys()])]

#         elif feature_importances == "LIME" and set_name == "test":
#             lime_result = list()
#             with open(f"E://AutoML//models//{one_model_name}//model.pkl", "rb") as f:
#                 one_model = pickle.load(f)
#             lime_explainer = lime_tabular.LimeTabularExplainer(training_data = trainData[input_features].values,
#                                                             training_labels = trainData[target_label].values,
#                                                             feature_names = input_features,
#                                                             class_names = ["Operation", "Biopsy"])
#             for test_index in tqdm(list(testData.index), desc = f"{one_model_name}-LIME"):
#                 try:
#                     exp = lime_explainer.explain_instance(testData[input_features].values[test_index, :], 
#                                                         predict_fn = one_model.model.predict_proba)
#                     exp.save_to_file(f"LIME_result//{one_model_name}_testData-{test_index}_OrininalResult-{int(testData.loc[test_index, target_label])}_HyperparameterTuning-{hyperparameter_tuning}.html")
#                     one_lime_result_dict_list = [{"Model": one_model_name, 
#                                                 "TestID": test_index,
#                                                 "TrueResult": int(testData.loc[test_index, target_label]),
#                                                 "PredResult": one_model.model.predict_proba(testData.loc[test_index, input_features].values.reshape((1, -1))), 
#                                                 "Information": i[0],
#                                                 "Value": i[1]} for i in exp.as_list()]    
#                     lime_result.extend(one_lime_result_dict_list)
#                 except: 
#                     pass
    
#     ### 將超參數調整流程結果作彙整 ###
#     hyperparameter_result = pd.concat(
#         [i["Hyperparameter_Tuning"] for i in predictor.values()], axis = 0
#     ).to_dict("records")
#     ### 將超參數調整流程結果作彙整 ###

#     ### 將超參數重要性的圖型作彙整 ###
#     param_plots = {
#         one_model_name: predictor[one_model_name]["Param_Importance"] for one_model_name in model_name_list
#     }
#     ### 將超參數重要性的圖型作彙整 ###

#     if feature_importances == "PermutationImportance":
#         return totalResult, totalFeatureImportanceResult, hyperparameter_result, param_plots
#     elif feature_importances == "LIME":
#         return totalResult, lime_result, hyperparameter_result, param_plots
#     else:
#         return totalResult, hyperparameter_result, param_plots