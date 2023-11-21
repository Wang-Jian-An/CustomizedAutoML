import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from imblearn.datasets import fetch_datasets
dataset = fetch_datasets()["thyroid_sick"]

raw_data = dataset["data"]
raw_target = dataset["target"]

n_clusters = 20
kmeans_cluster = KMeans(n_clusters = 20, n_init = "auto")
kmeans_cluster.fit(raw_data)

cluster_yhat = kmeans_cluster.predict(raw_data)

def imbalanced_processing_with_blocks_for_binary_class(
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

imbalanced_processed_data = list()
imbalanced_processed_target = list()
for one_cluster in range(n_clusters):
    cluster_index = np.where(cluster_yhat == one_cluster)[0]
    imputed_data, imputed_target = imbalanced_processing_with_blocks_for_binary_class(
        one_block_data = raw_data[cluster_index, :],
        one_block_target = raw_target[cluster_index]
    )
    imbalanced_processed_data.append(imputed_data)
    imbalanced_processed_target.append(imputed_target)
