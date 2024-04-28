from .__init__ import *

"""
1. 原始序列統計量
2. 各階差分後的統計量
3. 各個 Window Size 後的統計量
"""

def ML_fit(
    all_sequence_data: Union[pd.Series, List[List[Union[int, float]]]],
    max_differencing: Optional[int] = None,
    max_ma: Optional[int] = None,
    raw_sequence_statistics: Optional[Union[List[str], str]] = None,
    differencing_sequence_statistics: Optional[Union[List[str], str]] = None,
    ma_sequence_statistics: Optional[Union[List[str], str]] = None
) -> List[Union[int, float]]:

    """
    針對非深度學習、神經網路之訓練方法，針對時間序列資料提取出有用的特徵，過程包含：
    1. 取出原始序列的統計量
    2. 取出經過差分計算後的統計量
    3. 取出經過移動平均計算後的統計量
    """

    result = dict()

    # 若輸入序列為 pd.Series，則保留特徵名稱，把資料變成 List 型態
    if type(all_sequence_data) == pd.Series:
        feature_name = all_sequence_data.name
        all_sequence_data = all_sequence_data.tolist()

    else:
        feature_name = None

    # 若時序只有一個，則直接重複一個
    all_sequence_data = [
        i if i.__len__() > 1 else i * 2
        for i in all_sequence_data
    ]

    # 確認差階與移動平均 Window 的最大值
    theorical_max_differencing = check_max_size_for_differencing(
        all_sequence_data = all_sequence_data
    )
    theorical_max_ma_window_size = check_max_size_for_ma_window_size(
        all_sequence_data = all_sequence_data
    )
    if max_differencing and max_differencing < theorical_max_differencing:
        num_differencing = max_differencing

    else:
        num_differencing = theorical_max_differencing

    if max_ma and max_ma < theorical_max_ma_window_size:
        num_ma_window_size = max_ma

    else:
        num_ma_window_size = theorical_max_ma_window_size

    # 原始數據計算統計量
    if raw_sequence_statistics:
        if type(raw_sequence_statistics) == str:
            raw_sequence_statistics = [raw_sequence_statistics]
        
        result = {
            **result, 
            **{
                one_statistics: [
                    compute_statistics(
                        sequence_data = one_sequence_data,
                        statistics = one_statistics
                    )
                    for one_sequence_data in all_sequence_data
                ]
                for one_statistics in raw_sequence_statistics
            }
        }

    # 取出眾多差分序列
    if differencing_sequence_statistics:
        differencing_data = {
            one_of_num_differencing: compute_differencing(
                sequence_data = all_sequence_data,
                size_of_differencing = one_of_num_differencing
            )
            for one_of_num_differencing in range(1, num_differencing+1)
        }

        if type(differencing_sequence_statistics) == str:
            differencing_sequence_statistics = [differencing_sequence_statistics]

        # 計算眾多差分序列的統計量
        result = {
            **result, 
            **{
                f"diff_{one_of_num_differencing}_{one_statistics}": [
                    compute_statistics(
                        sequence_data = one_sequence_data,
                        statistics = one_statistics
                    )
                    for one_sequence_data in one_differencing_data
                ]
                for one_statistics, (one_of_num_differencing, one_differencing_data) in itertools.product(
                    differencing_sequence_statistics,
                    differencing_data.items()
                )
            }
        }

    # 取出眾多移動平均序列
    if ma_sequence_statistics:
        ma_data = {
            one_of_num_ma: compute_ma(
                sequence_data = all_sequence_data,
                size_of_ma_window_size = one_of_num_ma
            )
            for one_of_num_ma in range(1, num_ma_window_size+1)
        }

        # 計算眾多移動平均序列的統計量
        result = {
            **result, 
            **{
                f"ma_{one_of_num_ma}_{one_statistics}": [
                    compute_statistics(
                        sequence_data = one_sequence_data,
                        statistics = one_statistics
                    )
                    for one_sequence_data in one_ma_data
                ]
                for one_statistics, (one_of_num_ma, one_ma_data) in itertools.product(
                    differencing_sequence_statistics,
                    ma_data.items()
                )
            }
        }

    # 如果輸入型態是 pd.Series，則要加入特徵名稱
    if feature_name:
        result = {
            f"{feature_name}_{original_key}": value
            for original_key, value in result.items()
        }
    return result

def DL_fit():
    return 

def compute_statistics(
    sequence_data: List[Union[int, float]],
    statistics: str = "mean"
) -> Union[int, float]:

    """
    計算序列的統計量。
    """

    if statistics == "mean":
        return np.mean(sequence_data)

    elif statistics == "std":
        return np.std(sequence_data)

    elif statistics == "max":
        return max(sequence_data)

    elif statistics == "min":
        return min(sequence_data)

    elif statistics == "median":
        return np.median(sequence_data)
    
    elif statistics == "length":
        return sequence_data.__len__()

def compute_ma(
    sequence_data: Union[List[Union[int, float]], List[List[Union[int, float]]]],
    size_of_ma_window_size: int
) -> Union[int, float, List[Union[int, float]]]:
    
    """
    按照 Window size 大小，計算移動平均結果，
    假設有個序列是 [3, 1, 2, 5]，則 Window Size = 2 時等於 [2, 1.5, 3.5]
    """

    if type(sequence_data[0]) == list:        
        return [
                [
                compute_statistics(
                    sequence_data = j[i: i+size_of_ma_window_size],
                    statistics = "mean"
                )
                for i in range(j.__len__() - size_of_ma_window_size + 1)
            ] 
            for j in sequence_data
        ]

    else:
        return [
            compute_statistics(
                sequence_data = sequence_data[i: i+size_of_ma_window_size],
                statistics = "mean"
            )
            for i in range(sequence_data.__len__() - size_of_ma_window_size + 1)
        ]

def compute_differencing(
    sequence_data: Union[List[Union[int, float]], List[List[Union[int, float]]]],
    size_of_differencing: int
) -> Union[int, float, List[Union[int, float]]]:
    
    """
    按照階數，計算差分序列，
    假設有個序列是 [3, 1, 2, 5]，則二階差分等於 [2 - 3, 5 - 1]
    """

    if type(sequence_data[0]) == list:
        last_length = [
            i[size_of_differencing:].__len__()
            for i in sequence_data
        ]
        return [
            (
                np.array(i[size_of_differencing:]) - np.array(i[:one_last_length])
            ).tolist()
            for i, one_last_length in zip(
                sequence_data,
                last_length
            )
        ]

    else:
        last_length = sequence_data[size_of_differencing:].__len__()
        return (
            np.array(sequence_data[size_of_differencing:]) - np.array(sequence_data[:last_length])
        ).tolist()

def check_max_size_for_differencing(
    all_sequence_data: List[List[Union[int, float]]]
) -> int:

    """
    確認差分能夠達到最高是幾階，假設有個序列是 [3, 1, 2, 5]，則最多只能有三階差分，
    計算方法大致是序列長度減一。
    """

    return min([i.__len__() for i in all_sequence_data]) - 1

def check_max_size_for_ma_window_size(
    all_sequence_data: List[List[Union[int, float]]]
) -> int:

    """
    確認移動平均能夠達到最高的 Window Size 是多少，假設有個序列是 [3, 1, 2, 5]，則最多只能 Window Size = 4。    
    計算方式大致直接是等於序列長度。
    """

    return min([i.__len__() for i in all_sequence_data])