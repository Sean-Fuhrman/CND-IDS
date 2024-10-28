import numpy as np
from sklearn.cluster import KMeans
import logging
from kneed import KneeLocator
from collections import Counter, defaultdict

logger = logging.getLogger()

class K_Means():
    def __init__(self, datastream, f_e):
        self.datastream = datastream
        self.curr_experience = -1
        self.f_e = f_e
        logger.info('K_Means initialized')
        
    def fit(self, x) -> None:
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
        self.curr_experience += 1
        logger.info('K_Means fit')

    def getClusterLabels(self, x: np.ndarray) -> np.ndarray:
        return self.kmeans.predict(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        
        #Assign labels to clusters then predict x labels
        x_sample, y_sample = self.datastream.get_KMeans_subset(self.curr_experience)
        
        x_sample_clusters = self.kmeans.predict(self.f_e(x_sample))

        # Create a mapping from clusters to labels
        cluster_to_label = defaultdict(int)

        for cluster in np.unique(x_sample_clusters):
            # Find the labels of the points in the current cluster
            labels_in_cluster = y_sample[x_sample_clusters == cluster]
            
            # Determine the most common label in the cluster
            if len(labels_in_cluster) > 0:
                most_common_label = Counter(labels_in_cluster).most_common(1)[0][0]
                cluster_to_label[cluster] = most_common_label
        print(x.shape)
        print(x)
        print(np.isnan(x).sum())
        print(x.dtype)
        x_clusters = self.kmeans.predict(x)
        
        x_labels = np.array([cluster_to_label[cluster] for cluster in x_clusters])
        x_labels = (x_labels * 2) - 1
        return x_labels


