import numpy as np
from sklearn.decomposition import PCA as PCA
from typing import Union


class PCA_model():
    def __init__(self, pca_dim: Union[int, str] = 'auto', svd_solver: str = 'full'):
        self.pca_dim = pca_dim
        self.svd_solver = svd_solver
        self.pca = None
    def fit(self, x: np.ndarray, univariate: bool = False, verbose: bool = False) -> None:
        if self.pca_dim == 'auto':
            n_features = x.shape[1]
            if univariate:
                dim = 2
            elif n_features <= 50:
                dim = 10
            else:
                dim = 30

            if dim > x.shape[1]:
                # If the number of components is too large, then use a simple heuristic to select it.
                old_dim = dim
                dim = min(max(2, x.shape[1] // 5), x.shape[1])
                print(f'Adjusting estimated number of PCA components from {old_dim} to {dim}.')

            self.pca = PCA(n_components=dim, svd_solver=self.svd_solver)
        self.pca.fit(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        latent = self.pca.transform(x)
        reconstructed = self.pca.inverse_transform(latent)

        return np.abs(x - reconstructed).mean(axis=1)