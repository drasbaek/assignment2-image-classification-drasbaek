from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


'''
This file contains the model space for the model selection.
The model space is a dictionary with the model name as key and a dictionary with the model type and parameter grid as value.
New model types can be added to the model space by adding a new key-value pair to the dictionary.
These can then be used in the model selection for the classify.py script.
'''

model_space = {
    "logistic": {
        "type": LogisticRegression(max_iter = 1000),
        "param_grid": {
            "C": [0.001, 0.01, 0.1, 1],
            "tol": [0.01, 0.1, 1],
            "penalty": ["l1", "l2"],
            "solver": ["saga"],
            "multi_class": ["multinomial"]
        }
    },
    "neural": {
        "type": MLPClassifier(max_iter = 1000),
        "param_grid": {
            "hidden_layer_sizes": [(512), (128), (32), (128,32), (512,128,32)],
            "learning_rate": ["adaptive"],
            "early_stopping": [True]
        }
    },
    "random_forest": {
        "type": RandomForestClassifier(),
        "param_grid": {
            "n_estimators": [10, 20, 30],
            "max_depth": [5, 10, 15]
        }
    }
}
