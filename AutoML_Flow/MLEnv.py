import itertools

standardization_list = ["None", "standardization", "normalization", "min-max_scaler"]
feature_selection_method_list = ["None", "SFS", "SBS", "SFFS", "SFBS"]
decomposition_list = ["None", "PCA", "IPCA"]
imbalanced_list = ["None", "SMOTE"]
totalMLMethodsList = list(set(
    standardization_list + feature_selection_method_list + decomposition_list + imbalanced_list
))  
featureEngineerFlow = [
    {
        "Imbalanced": oneBalanced,
        "Decomposition": oneDecomposition,
        "Standardization": oneStandardization,
        "FeatureSelection": oneFeatureSelection
    } for oneBalanced, oneDecomposition, oneStandardization, oneFeatureSelection in itertools.product(
        imbalanced_list,
        decomposition_list,
        standardization_list,
        feature_selection_method_list
    )
]