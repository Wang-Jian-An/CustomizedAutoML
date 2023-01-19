import numpy as np
import pandas as pd
from itertools import combinations

def polynomial_kernel_function_with_degree_two(data_1: np.ndarray, 
                                               data_2: np.ndarray,
                                               c: int or float = 0) -> list:
    
    # 確認 x, y 皆為一維向量且長度相同
    assert np.ndim(data_1) == 1, "The dimension of x must be 1. "
    assert np.ndim(data_2) == 1, "The dimension of y must be 1. "
    assert data_1.shape[0] == data_2.shape[0] == 0, "The length of x and y must same. "
    
    newData_1 = [*np.power(data_1, 2).tolist(), *(np.square(2)*np.array([data_1[i]*data_2[j] for i, j in combinations(range(data_1.shape[0]), 2)]) )]
    newData_2 = [*np.power(data_2, 2).tolist(), *(np.square(2)*np.array([data_1[i]*data_2[j] for i, j in combinations(range(data_1.shape[0]), 2)]) )]
    return newData_1, newData_2

def executeKernelFunc(data: pd.DataFrame, 
                      inputFeatures: list):
    return 