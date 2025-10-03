from sklearn.neural_network import MLPClassifier

class mlp_classifier:
    def __init__(self, X, y):
        params = {
            # "hidden_layer_sizes": (30, 10, ),
            # "solver": "adam",
            # "activation": "relu",
            # "alpha": 1e-4,
            "max_iter": 5000,
        }
        self._model = MLPClassifier(**params)
        self._model.fit(X, y)
    
    @property
    def fitted_model(self):
        return self._model
    
    def predict(self, X):
        return self._model.predict(X)
    
    def predict_proba(self, X):
        return self._model.predict_proba(X)