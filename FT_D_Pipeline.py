import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA

class ML_Pipeline():
    def __init__(self, ml_methods: str or list, inputFeatures, target):
        methods_dict = {
            "None": None,
            "standardization": StandardScaler(),
            "normalization": Normalizer(),
            "min-max_scaler": MinMaxScaler(),
            "PCA": PCA(),
            "KernelPCA": KernelPCA(),
            "IPCA": IncrementalPCA()
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
                if method_obj is None:
                    continue
                assert type(fit_data) == pd.DataFrame, "The variable 'fit_data' must be a DataFrame. "
                self.each_flow_input_features[method_name] = self.inputFeatures
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
                           transform_data: pd.DataFrame):
        # 若沒有做任一特徵工程，則可不必運行此 Pipeline
        if self.ML_flow_obj is None or all([i is None for i in self.ML_flow_obj]):
            return transform_data
        else:
            # 輪流執行特徵轉換或降維
            for method_name, method_obj in self.ML_flow_obj.items():

                if method_obj is not None:
                    transform_data = pd.concat([
                        pd.DataFrame(method_obj.transform(transform_data[self.each_flow_input_features[method_name]].values), columns = self.each_flow_output_features[method_name]),
                        transform_data[self.target]
                    ], axis = 1)
            return transform_data