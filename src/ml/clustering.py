"""
K-Means clustering implementation.

Provides wrapper around scikit-learn K-Means with metrics calculation
and visualization support.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, Optional

from src.core.config import Config


class KMeansClustering:
    """
    K-Means clustering for sentence embeddings.

    Wraps scikit-learn K-Means with configuration management and
    automatic metrics calculation. Fixed at k=3 clusters to match
    the three sentence categories.

    Attributes:
        n_clusters: Number of clusters (fixed at 3)
        kmeans: Scikit-learn KMeans model
        labels_: Cluster labels after fitting
        cluster_centers_: Cluster centroids after fitting
        inertia_: Sum of squared distances to nearest cluster center

    Example:
        >>> clustering = KMeansClustering(config)
        >>> labels = clustering.fit(vectors)
        >>> metrics = clustering.get_metrics(vectors)
        >>> print(metrics['silhouette_score'])
        0.742
    """

    def __init__(self, config: Config):
        """
        Initialize K-Means clustering with configuration.

        Args:
            config: Configuration object with clustering parameters
        """
        self.n_clusters = config.kmeans_clusters  # Should be 3

        # Initialize K-Means with parameters optimized for quality
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",  # Smart initialization
            max_iter=300,  # Maximum iterations
            n_init=10,  # Number of different initializations
            random_state=config.random_state,  # For reproducibility
            algorithm="lloyd",  # Standard K-Means algorithm
        )

        # Will be set after fitting
        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None

    def fit(self, vectors: np.ndarray) -> np.ndarray:
        """
        Fit K-Means clustering on vectors.

        Runs the K-Means algorithm and stores the results internally.
        Returns cluster labels for convenience.

        Args:
            vectors: NumPy array of shape (n_samples, n_features)

        Returns:
            Cluster labels array of shape (n_samples,) with values 0, 1, or 2

        Raises:
            ValueError: If vectors is not 2D array or has wrong shape

        Example:
            >>> labels = clustering.fit(training_vectors)
            >>> labels.shape
            (30,)
            >>> set(labels)
            {0, 1, 2}
        """
        # Validate input
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {vectors.shape}")

        if len(vectors) < self.n_clusters:
            raise ValueError(
                f"Need at least {self.n_clusters} samples, got {len(vectors)}"
            )

        # Fit K-Means
        self.kmeans.fit(vectors)

        # Store results
        self.labels_ = self.kmeans.labels_
        self.cluster_centers_ = self.kmeans.cluster_centers_
        self.inertia_ = self.kmeans.inertia_

        return self.labels_

    def get_cluster_centers(self) -> np.ndarray:
        """
        Get cluster centroids.

        Returns:
            Array of shape (n_clusters, n_features) with cluster centers

        Raises:
            RuntimeError: If fit() hasn't been called yet
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Must call fit() before getting cluster centers")

        return self.cluster_centers_

    def get_metrics(
        self, vectors: np.ndarray, true_labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate clustering quality metrics.

        Computes:
        - Silhouette score: Measures how similar samples are to their own
          cluster vs other clusters. Range [-1, 1], higher is better.
        - Inertia: Sum of squared distances to nearest cluster center.
          Lower is better.
        - Cluster sizes: Number of samples in each cluster
        - ARI: Adjusted Rand Index comparing to true labels
        - Purity: Percentage of correctly clustered samples

        Args:
            vectors: Original vectors used for clustering
            true_labels: Optional true category labels for purity and ARI

        Returns:
            Dictionary with metric names and values

        Example:
            >>> metrics = clustering.get_metrics(vectors, true_labels)
            >>> print(f"Silhouette: {metrics['silhouette_score']:.3f}")
            >>> print(f"ARI: {metrics['ari']:.3f}")
        """
        from sklearn.metrics import adjusted_rand_score

        if self.labels_ is None:
            raise RuntimeError("Must call fit() before calculating metrics")

        # Calculate silhouette score (measures cluster quality)
        # Range: [-1, 1], higher is better
        # >0.7 = strong clusters, >0.5 = reasonable, <0.25 = weak
        sil_score = silhouette_score(vectors, self.labels_)

        # Get cluster sizes
        cluster_sizes = np.bincount(self.labels_).tolist()

        metrics = {
            "silhouette_score": float(sil_score),
            "inertia": float(self.inertia_),
            "cluster_sizes": cluster_sizes,
            "n_clusters": self.n_clusters,
        }

        # Add purity and ARI if true labels provided
        if true_labels is not None:
            purity = self._calculate_purity(true_labels)
            ari = adjusted_rand_score(true_labels, self.labels_)
            metrics["purity"] = float(purity)
            metrics["ari"] = float(ari)

        return metrics

    def _calculate_purity(self, true_labels: np.ndarray) -> float:
        """
        Calculate cluster purity score.

        Purity measures how "pure" each cluster is - i.e., what percentage
        of samples in each cluster belong to the same true class.

        Args:
            true_labels: True category labels

        Returns:
            Purity score in range [0, 1], higher is better
        """
        # For each cluster, find the most common true label
        total_correct = 0

        for cluster_id in range(self.n_clusters):
            # Get true labels of samples in this cluster
            cluster_mask = self.labels_ == cluster_id
            cluster_true_labels = true_labels[cluster_mask]

            if len(cluster_true_labels) > 0:
                # Count most common label
                most_common_count = np.bincount(cluster_true_labels).max()
                total_correct += most_common_count

        # Purity = total correct assignments / total samples
        purity = total_correct / len(true_labels)
        return purity
