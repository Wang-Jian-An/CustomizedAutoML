import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from .Model_Prediction import modelPrediction

class tabular_LIME:
    def __init__(
        self,
        original_model: list,
        target_label: str,
        features_list: list
    ):
        
        """
        Args: 
        - original_model (list): Many models based on scikit-learn
        - target_label (str)
        - features_list (list)
        """

        if not(type(original_model) == list):
            original_model = [original_model]

        self.original_model = original_model
        self.target_label_dict = {
            index: i
            for index, i in enumerate(target_label)
        }
        self.features_list = features_list
        return
    
    def fit(
        self,
        explain_instance: dict | pd.Series | pd.DataFrame
    ):

        """
        Args: 
        - explain_instance (dict or pd.Series or pd.DataFrame)
        """

        # Step0. Clean data
        if isinstance(explain_instance, pd.Series):
            explain_instance = explain_instance.to_dict()
        elif isinstance(explain_instance, pd.DataFrame):
            assert explain_instance.shape[0] == 1, "There is only one instance to explain. "
            explain_instance = explain_instance.to_dict("records")[0]

        # Step1. Define some of the binary vector
        binary_vectors = self.define_binary_vector(
            num_features = explain_instance.keys().__len__()
        )

        # Step2. Generate data via binary vector
        simulation_data = self.generate_data(
            explain_data = explain_instance,
            binary_vectors = binary_vectors
        )

        # Step3. Optimize loss function to define explanation model
        explanation_model, self.explain_instance_label = self.optimize_loss(
            explain_data = explain_instance,
            simulation_data = simulation_data
        )

        # Step4. Convert the weights of explanation model into pd.Series
        self.explanation_weights = pd.Series(
            explanation_model,
            index = list(explain_instance.keys())
        )
        return 

    def output_explanation_model_weight(
        self
    ) -> pd.Series:
        return self.explanation_weights

    def draw_forest_plot(
        self
    ) -> matplotlib.axes.Axes:
        
        """
        <Explanation TBD>
        """

        fig = plt.figure(figsize = (19.20, 10.80))
        plt.subplot(121)
        sns.barplot(
            x = self.original_data_proba,
            y = self.target_label_dict.values(),
        )
        plt.title("The probability for each class", fontsize = 24)
        plt.ylabel("Label", fontsize = 20)
        plt.xlabel("Probability", fontsize = 20)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        
        plt.subplot(122)
        self.explanation_weights.plot.barh()
        plt.title("The importance for each features", fontsize = 24)
        plt.ylabel("Features", fontsize = 20)
        plt.xlabel("Importance", fontsize = 20)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        plt.tight_layout()
        return fig

    def define_binary_vector(
        self,
        num_features: int,
        num_binary_vector_ratio: float | None = None,
        binary_ratio: float = 0.5
    ):
        
        """
        <Explanation TBD>
        """

        if not(num_binary_vector_ratio):
            num_binary_vector_ratio = 1 / np.power(2, num_features * 0.5)

        return np.random.binomial(
            n = num_features,
            p = binary_ratio,
            size = (int(np.power(2, num_features) * num_binary_vector_ratio), num_features)
        )

    def generate_data(
        self,
        explain_data: dict,
        binary_vectors: np.ndarray
    ):
        
        """
        <Explanation TBD>
        """

        binary_vectors_plus_random_value = np.where(
            binary_vectors == 0,
            np.random.normal(loc = 0, scale = 1),
            binary_vectors            
        )

        explain_features = list(explain_data.keys())
        explain_data = np.array([list(explain_data.values())] * binary_vectors.shape[0])
        explain_data = explain_data + binary_vectors_plus_random_value
        return pd.DataFrame(
            explain_data,
            columns = explain_features
        )

    def optimize_loss(
        self,
        explain_data: dict, 
        simulation_data: pd.DataFrame
    ):
        
        """
        <Explanation TBD>
        """

        def objective(
            explanation_model: np.ndarray,
            simulation_data: pd.DataFrame, 
            simulation_data_proba: np.ndarray,
            proximity: np.ndarray
        ):
            
            simulation_data = simulation_data.values
            explanation_label = np.matmul(simulation_data, explanation_model[:, np.newaxis]).flatten()
            loss = np.sum(
                proximity * np.power(simulation_data_proba - explanation_label, 2)
            )
            return loss

        original_data_result = modelPrediction(
            modelList = self.original_model,
            targetType = "classification",
            featureList = self.features_list,
            predData = pd.DataFrame.from_records([explain_data])
        )
        original_data_label = original_data_result["Yhat"][0] # 原始資料之答案
        self.original_data_proba = np.array(original_data_result["YhatProba"]).flatten() # 原始資料之機率值
        explain_data = np.array(
            [list(explain_data.values())] * simulation_data.shape[0]
        )
        proximity = self.compute_proximity(
            explain_data = explain_data,
            simulation_data = simulation_data
        )
        simulation_data_result = modelPrediction(
            modelList = self.original_model,
            targetType = "classification",
            featureList = self.features_list,
            predData = simulation_data
        )
        simulation_data_proba = np.array(simulation_data_result["YhatProba"]) # 模擬資料答案之機率值
        explanation_model = self.initial_explanation_model(num_features = explain_data.shape[1])
        optim = minimize(
            fun = objective,
            x0 = explanation_model,
            args = (
                simulation_data,
                simulation_data_proba[:, original_data_label],
                proximity
            )
        )
        return optim.x, original_data_label

    def fit_explanation_model(
        self,
        X: np.ndarray,
        weight: np.ndarray
    ) -> np.ndarray:
        
        """
        <Explanation TBD>
        """

        
        return 

    def compute_distance(
        self,
        explain_data: np.ndarray,
        simulation_data: np.ndarray,
        method: str = "euclidean-distance"
    ) -> np.ndarray:

        """
        <Explanation TBD>
        """

        if method == "euclidean-distance":
            return np.sqrt(
                np.sum(
                    np.power(np.divide(explain_data - simulation_data, explain_data), 2),
                    axis = 1
                )                
            )
        else:
            return np.sqrt(
                np.sum(
                    np.power(np.divide(explain_data - simulation_data, explain_data), 2),
                    axis = 1
                )                
            )
    
    def initial_explanation_model(
        self,
        num_features: int
    ):
        
        """
        Initialize linear model for explanation. 

        Args: 
            - num_features: The number of features. 
        Returns: 
            - weights: The weights of linear model. 
        """

        return np.random.normal(loc = 0, scale = 1, size = num_features)
    
    def compute_proximity(
        self,
        explain_data: np.ndarray,
        simulation_data: np.ndarray
    ) -> np.ndarray:
        
        """
        <Explanation TBD>
        """

        distance_vector = self.compute_distance(
            explain_data = explain_data,
            simulation_data = simulation_data
        )
        return np.exp(
            -1 * np.power(distance_vector, 2) / (distance_vector.shape[0] ** 2)
        )