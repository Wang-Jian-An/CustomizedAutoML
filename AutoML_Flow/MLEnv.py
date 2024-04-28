from .__init__ import *

with open("./config.yaml") as f:
    ml_config = yaml.safe_load(f)

totalMLMethodsList = list(set(
    ml_config["standardization"] + ml_config["feature_selection"] + ml_config["decomposition"] + ml_config["imbalanced"]
))  
featureEngineerFlow = [
    {
        "FeatureSelection": oneFeatureSelection, 
        "Imbalanced": oneBalanced,
        "Decomposition": oneDecomposition,
        "Standardization": oneStandardization,
    } for oneFeatureSelection, oneBalanced, oneDecomposition, oneStandardization in itertools.product(
        ml_config["feature_selection"],
        ml_config["imbalanced"],
        ml_config["decomposition"],
        ml_config["standardization"],
        
    )
]