import os
import yaml
import itertools

with open(os.path.join(os.path.dirname(__file__), "./config.yaml")) as f:
    ml_config = yaml.safe_load(f)

totalMLMethodsList = list(set(
    ml_config["standardization"] + ml_config["feature_selection"] + ml_config["decomposition"] + ml_config["imbalanced"] + ml_config["regression_transform"]
))  
featureEngineerFlow = [
    {
        "FeatureSelection": oneFeatureSelection, 
        "Imbalanced": oneBalanced,
        "Decomposition": oneDecomposition,
        "Standardization": oneStandardization,
        "Regression_transform": one_regression_transform
    } for oneFeatureSelection, oneBalanced, oneDecomposition, oneStandardization, one_regression_transform in itertools.product(
        ml_config["feature_selection"],
        ml_config["imbalanced"],
        ml_config["decomposition"],
        ml_config["standardization"],
        ml_config["regression_transform"]
    )
]
model_combination = [
    *itertools.combinations(ml_config["regression_model"], 1),
    *itertools.combinations(ml_config["regression_model"], 2),
    *itertools.combinations(ml_config["regression_model"], 3),
    *itertools.combinations(ml_config["regression_model"], 4),
    *itertools.combinations(ml_config["regression_model"], 5),
]