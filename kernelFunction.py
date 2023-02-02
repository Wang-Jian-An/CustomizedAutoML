import numpy as np
import pandas as pd
from itertools import combinations

def polynomial_kernel_function_with_degree_two(one_data: np.ndarray,
                                               c: int or float = 0) -> list:
    
    # 確認 x, y 皆為一維向量且長度相同
    assert np.ndim(one_data) == 2, "The dimension of x must be 1. "
    return np.array([
        np.power(one_data, 2).T.tolist(),
        np.array([one_data[:, i] * one_data[:, j] for i, j in combinations(list(range(one_data.shape[1])), 2)]).tolist()
    ]).T

def executeKernelFunc(data: pd.DataFrame, 
                      inputFeatures: list):
    return 