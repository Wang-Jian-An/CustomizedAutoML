import itertools

standardization_list = ["None", "standardization", "normalization", "min-max_scaler"]
feature_selection_method_list = ["None", "SFS", "SBS", "SFFS", "SFBS"]
decomposition_list = ["None", "PCA", "IPCA"]
imbalanced_list = ["None", "SMOTE", "SMOTEEEN", "SMOTETomek"]
totalMLMethodsList = list(set(
    standardization_list + feature_selection_method_list + decomposition_list + imbalanced_list
))  
featureEngineerFlow = [
    {
        "FeatureSelection": oneFeatureSelection, 
        "Imbalanced": oneBalanced,
        "Decomposition": oneDecomposition,
        "Standardization": oneStandardization,
    } for oneFeatureSelection, oneBalanced, oneDecomposition, oneStandardization in itertools.product(
        feature_selection_method_list,
        imbalanced_list,
        decomposition_list,
        standardization_list,
        
    )
]