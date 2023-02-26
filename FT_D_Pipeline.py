import numpy as np
import pandas as pd
import itertools
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from imblearn.over_sampling import SMOTE

class ML_Pipeline():
    def __init__(self, ml_methods: str or list, inputFeatures, target):
        assert ml_methods[0] == "SMOTE" if "SMOTE" in ml_methods else True, "SMOTE is must the first method. "
        methods_dict = {
            "None": None,
            "standardization": StandardScaler(),
            "normalization": Normalizer(),
            "min-max_scaler": MinMaxScaler(),
            "PCA": PCA(),
            "KernelPCA": KernelPCA(),
            "IPCA": IncrementalPCA(),
            "Poly-Kernel": self.executeKernelFunc,
            "SMOTE": SMOTE()
        }
        self.ML_flow_obj = {
            i: methods_dict[i] for i in ml_methods
        } if type(ml_methods) == list else {
            ml_methods: methods_dict[ml_methods]
        }
        self.each_flow_input_features = {i: None for i in ml_methods} if type(ml_methods) == list else {ml_methods: None}
        self.each_flow_output_features = {i: None for i in ml_methods} if type(ml_methods) == list else {ml_methods: None}
        self.inputFeatures = inputFeatures
        self.target = target
        return 

    def fit_Pipeline(self, 
                     fit_data: pd.DataFrame,  
                    decomposition_result_file_name: str = None):
        if self.ML_flow_obj is None or all([i is None for i in self.ML_flow_obj]):
            return 
        else:       
            for method_name, method_obj in self.ML_flow_obj.items():
                if method_obj is None or method_name == "SMOTE":
                    continue
                assert type(fit_data) == pd.DataFrame, "The variable 'fit_data' must be a DataFrame. "

                self.each_flow_input_features[method_name] = self.inputFeatures
                if method_name == "Poly-Kernel":
                    fit_data =  method_obj(data = fit_data, 
                                            inputFeatures = self.each_flow_input_features[method_name],
                                            target = self.target)
                    self.inputFeatures = fit_data.drop(columns = self.target).columns.tolist()
                    self.each_flow_output_features[method_name] = self.inputFeatures
                    continue

                method_obj.fit(fit_data[self.inputFeatures].values, fit_data[self.target].values)
                if method_name in ["PCA", "IPCA"] and decomposition_result_file_name:
                    pd.DataFrame(method_obj.components_.T, 
                                 index = self.each_flow_input_features[method_name], 
                                 columns = method_obj.get_feature_names_out().tolist()).to_excel(decomposition_result_file_name)  
                    cumsum_var_ratio = np.cumsum(method_obj.explained_variance_ratio_)
                    select_num_of_cumsum_var_ratio = np.where(cumsum_var_ratio < 0.9)[0].shape[0]
                    method_obj.set_params(**{"n_components": select_num_of_cumsum_var_ratio})
                    method_obj.fit(fit_data[self.each_flow_input_features[method_name]].values, fit_data[self.target].values)
                    self.inputFeatures = method_obj.get_feature_names_out().tolist()
                self.each_flow_output_features[method_name] = self.inputFeatures
                fit_data = pd.concat([
                    pd.DataFrame(method_obj.transform(fit_data[self.each_flow_input_features[method_name]].values), columns = self.each_flow_output_features[method_name]),
                    fit_data[self.target]
                ], axis = 1)
            return
    
    def transform_Pipeline(self, 
                           transform_data: pd.DataFrame,
                           mode: str): 
        # 若沒有做任一特徵工程，則可不必運行此 Pipeline
        if self.ML_flow_obj is None or all([i is None for i in self.ML_flow_obj]):
            return transform_data
        else:
            # 輪流執行特徵轉換或降維
            for method_name, method_obj in self.ML_flow_obj.items():
                if mode == "train" and method_name == "SMOTE":
                    X_res, y_res = method_obj.fit_resample(transform_data[self.inputFeatures].values, transform_data[self.target].values)
                    transform_data = pd.concat(
                        [pd.DataFrame(X_res, columns = self.inputFeatures), pd.Series(y_res, name = self.target)], axis = 1
                    )
                elif method_name == "Poly-Kernel":
                    transform_data =  method_obj(data = transform_data, 
                                                inputFeatures = self.each_flow_input_features[method_name],
                                                target = self.target)
                elif method_obj is not None and method_name != "SMOTE":
                    transform_data = pd.concat([
                        pd.DataFrame(method_obj.transform(transform_data[self.each_flow_input_features[method_name]].values), columns = self.each_flow_output_features[method_name]),
                        transform_data[self.target]
                    ], axis = 1)
            return transform_data 

    def polynomial_kernel_function_with_degree_two(self, 
                                                   one_data: np.ndarray,
                                                   c: int or float = 0) -> list:
        # 確認 x, y 皆為一維向量且長度相同
        assert np.ndim(one_data) == 2, "The dimension of x must be 1. "
        return np.array([
            *np.power(one_data, 2).T.tolist(),
            *np.array([one_data[:, i] * one_data[:, j] for i, j in itertools.combinations(list(range(one_data.shape[1])), 2)]).tolist()
        ]).T 

    def executeKernelFunc(self, 
                        data: pd.DataFrame, 
                        inputFeatures: list,
                        target: str):
        kernel_data = self.polynomial_kernel_function_with_degree_two(one_data = data[inputFeatures].values)
        kernel_inputFeatures = [f"{i}_degree_2" for i in inputFeatures] + [f"{i}_{j}" for i, j in itertools.combinations(inputFeatures, 2)]
        data = pd.concat([
            pd.DataFrame(kernel_data, columns = kernel_inputFeatures),
            data[target]
        ], axis = 1)                  
        return data