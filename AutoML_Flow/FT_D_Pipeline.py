import numpy as np
import pandas as pd
import itertools
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

class ML_Pipeline():
    def __init__(
        self, 
        ml_methods: str or list, 
        inputFeatures, 
        target
    ):
        methods_dict = {
            "None": None,
            "standardization": StandardScaler(),
            "normalization": Normalizer(),
            "min-max_scaler": MinMaxScaler(),
            "PCA": PCA(),
            "KernelPCA": KernelPCA(),
            "IPCA": IncrementalPCA(),
            "Poly-Kernel": self.executeKernelFunc,
            "SMOTE": SMOTE(),
            "SMOTEENN": SMOTEENN(),
            "SMOTETomek": SMOTETomek(), 
            "SMOTE_block": self.imputation_for_block
        }
        if type(ml_methods) == list:
            self.ML_flow_obj = {
                i: methods_dict[i]
                for i in ml_methods if i in list(methods_dict.keys())
            }
        else:
            if ml_methods in list(methods_dict.values()):
                self.ML_flow_obj = {ml_methods: methods_dict[ml_methods]}
            else:
                self.ML_flow_obj = {"None": methods_dict["None"]}
        self.each_flow_input_features = {i: None for i in ml_methods} if type(ml_methods) == list else {ml_methods: None}
        self.each_flow_output_features = {i: None for i in ml_methods} if type(ml_methods) == list else {ml_methods: None}
        self.inputFeatures = inputFeatures
        self.target = target
        return 

    def fit_Pipeline(self, 
                     fit_data: pd.DataFrame,  
                    decomposition_result_file_name: str = None):
        """
        fit_data (pd.DataFrame): Data including input features. 
        """
        if self.ML_flow_obj is None or all([i is None for i in self.ML_flow_obj]):
            return 
        else:       
            for method_name, method_obj in self.ML_flow_obj.items():
                if method_name == "SMOTE_block":
                    self.n_clusters = int(fit_data.shape[0] / 15)
                    normalize_data = fit_data / fit_data.max()
                    self.kmeans_cluster = KMeans(n_clusters = self.n_clusters, n_init = "auto")
                    self.kmeans_cluster.fit(normalize_data[self.inputFeatures])

                if method_obj is None or method_name in ["SMOTE", "SMOTEENN", "SMOTETomek", "SMOTE_block"]:
                    continue
                assert type(fit_data) == pd.DataFrame, "The variable 'fit_data' must be a DataFrame. "

                self.each_flow_input_features[method_name] = self.inputFeatures
                if method_name == "Poly-Kernel":
                    fit_data =  method_obj(data = fit_data, 
                                            inputFeatures = self.each_flow_input_features[method_name])
                    self.inputFeatures = fit_data.columns.tolist()
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
                    method_obj.fit(fit_data[self.each_flow_input_features[method_name]].values)
                    self.inputFeatures = method_obj.get_feature_names_out().tolist()
                self.each_flow_output_features[method_name] = self.inputFeatures
                fit_data[self.inputFeatures] = method_obj.transform(fit_data[self.each_flow_input_features[method_name]].values)      
            return
    
    def transform_Pipeline(self, 
                           transform_data: pd.DataFrame,
                           transform_target: pd.Series, 
                           mode: str): 
        
        # 若沒有做任一特徵工程，則可不必運行此 Pipeline
        if self.ML_flow_obj is None or all([i is None for i in self.ML_flow_obj]):
            return transform_data, transform_target
        else:
            # 輪流執行特徵轉換或降維
            for method_name, method_obj in self.ML_flow_obj.items():
                if mode == "train" :
                    if method_name in ["SMOTE", "SMOTEENN", "SMOTETomek"]:
                        X_res, y_res = method_obj.fit_resample(transform_data[self.inputFeatures].values, transform_target)
                        transform_data = pd.DataFrame(X_res, columns = self.inputFeatures)
                        transform_target = pd.Series(y_res, name = self.target)
                    elif method_name == "SMOTE_block":
                        # print(transform_data.shape, transform_target.shape)
                        X_res, y_res = method_obj(transform_data[self.inputFeatures].values, transform_target.values.flatten())
                        transform_data = pd.DataFrame(np.vstack([i for i in X_res if i is not None]), columns = self.inputFeatures)
                        transform_target = pd.Series(np.hstack([i for i in y_res if i is not None]), name = self.target)
                        print(transform_target.value_counts())
                elif method_name == "Poly-Kernel":
                    transform_data =  method_obj(data = transform_data, 
                                                inputFeatures = self.each_flow_input_features[method_name])
                elif method_obj is not None and method_name not in ["SMOTE", "SMOTEENN", "SMOTETomek", "SMOTE_block"]:
                    transform_data = pd.DataFrame(
                        method_obj.transform(transform_data[self.each_flow_input_features[method_name]].values), 
                        columns = self.each_flow_output_features[method_name]
                    )
            return transform_data, transform_target

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
                        inputFeatures: list):
        kernel_data = self.polynomial_kernel_function_with_degree_two(one_data = data[inputFeatures].values)
        kernel_inputFeatures = [f"{i}_degree_2" for i in inputFeatures] + [f"{i}_{j}" for i, j in itertools.combinations(inputFeatures, 2)]
        data = pd.DataFrame(kernel_data, columns = kernel_inputFeatures)
        return data
    
    def imputation_for_block(
        self,
        data: np.ndarray or pd.DataFrame,
        target: np.ndarray or pd.Series
    ):

        normalize_data = data / data.max()
        cluster_yhat = self.kmeans_cluster.predict(normalize_data) 
        print("Cluster yhat", cluster_yhat)
        
        imbalanced_processed_data = list()
        imbalanced_processed_target = list()
        for one_cluster in range(self.n_clusters):
            cluster_index = np.where(cluster_yhat == one_cluster)[0]
            print("Cluster index", cluster_index)
            imputed_data, imputed_target = self.imbalanced_processing_with_blocks_for_binary_class(
                one_block_data = data[cluster_index, :],
                one_block_target = target[cluster_index]
            )
            imbalanced_processed_data.append(imputed_data)
            imbalanced_processed_target.append(imputed_target)
        return imbalanced_processed_data, imbalanced_processed_target
    
    def imbalanced_processing_with_blocks_for_binary_class(
        self, 
        one_block_data: np.ndarray or pd.DataFrame,
        one_block_target: np.ndarray or pd.Series,
        target_name: str = None
    ):

        """
        Compute the number of each class and its ratio. If the ratio is smaller than 5%, delete these data
        
        
        """

        init_target_type_array = False
        init_data_type_dataframe = False
        if type(one_block_target) == np.ndarray:
            one_block_target = pd.Series(one_block_target.flatten())
            init_target_type_array = True

        number_one_block_target = one_block_target.value_counts()
        number_each_class = sorted(number_one_block_target.tolist())
        print(number_one_block_target)
        if number_each_class.__len__() == 1:
            if init_target_type_array: # 如果兩類別的比例小於 5%，直接刪除資料
                one_block_target = one_block_target.values
            return one_block_data, one_block_target
        else: # 否則就使用 SMOTE
            class_ratio = number_each_class[0] / number_each_class[1]
            if class_ratio <= 0.05 or number_each_class[0] < 10: 
                return None, None
            else:
                if type(one_block_data) == pd.DataFrame:
                    data_columns = one_block_data.columns.tolist()
                    one_block_data = one_block_data.values
                    init_data_type_dataframe = True
                imputation = SMOTE()
                impute_data, impute_target = imputation.fit_resample(one_block_data, one_block_target.values)
                if init_data_type_dataframe:
                    impute_data = pd.DataFrame(
                        impute_data,
                        columns = data_columns
                    )
                if init_target_type_array == False:
                    impute_target = pd.Series(impute_target, name = target_name)
                return impute_data, impute_target