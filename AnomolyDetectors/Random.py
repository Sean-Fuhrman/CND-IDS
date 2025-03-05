import numpy as np

class Random():
    def __init__(self):
        pass

    def fit(self, init_normal) -> None:
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.random.rand(x.shape[0])