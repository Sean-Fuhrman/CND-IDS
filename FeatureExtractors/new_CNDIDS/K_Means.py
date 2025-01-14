
import numpy as np
from sklearn.cluster import KMeans
import logging
from kneed import KneeLocator
from collections import Counter, defaultdict
import warnings
logger = logging.getLogger()

class K_Means():
    def __init__(self):
        logger.info('K_Means initialized')
    

    def fit(self, x) -> None:
        #Suppress warnings
        warnings.filterwarnings("ignore")
        logger.info('Fitting K_Means')
        n_cluster_options = [3, 5, 7, 9, 11, 15]
        wcss = []
        for i in n_cluster_options:
            kmeans = KMeans(n_clusters=i,random_state=42)
            kmeans.fit(x)
            wcss.append(kmeans.inertia_)

        # Use the KneeLocator to find the elbow point
        kneedle = KneeLocator(n_cluster_options, wcss, curve='convex', direction='decreasing')
        optimal_n_clusters = kneedle.elbow
        if optimal_n_clusters is None:
            optimal_n_clusters = n_cluster_options[-1]
        self.kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
        self.kmeans.fit(x)
        logger.info('K_Means fit')

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.kmeans.predict(x)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.kmeans.predict(x)
 