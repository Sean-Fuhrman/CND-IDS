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
            pca = PCA()
            pca.fit(x)
            cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
            dim = np.argmax(cumulative_explained_variance >= 0.95) + 1
        else:
            dim = self.pca_dim
        
        self.pca = PCA(n_components=dim, svd_solver=self.svd_solver)
        # if x is tensor, convert to numpy
        if hasattr(x, 'numpy'):
            x = x.cpu().numpy()
        self.pca.fit(x)
        self.is_fit = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fit:
            raise ValueError('PCA model has not been fit yet.')
        if hasattr(x, 'numpy'):
            x = x.cpu().numpy()
        latent = self.pca.transform(x)
        reconstructed = self.pca.inverse_transform(latent)

        return np.abs(x - reconstructed).mean(axis=1)