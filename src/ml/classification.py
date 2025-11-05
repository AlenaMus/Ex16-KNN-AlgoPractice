"""
K-Nearest Neighbors classification implementation.

Provides wrapper around scikit-learn KNN classifier with metrics
calculation and prediction support.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Optional

from src.core.config import Config


class KNNClassifier:
    """
    K-Nearest Neighbors classifier for sentence categorization.

    Wraps scikit-learn KNeighborsClassifier with configuration management
    and automatic metrics calculation. Uses distance-weighted voting for
    better performance with normalized vectors.

    Attributes:
        n_neighbors: Number of neighbors to consider (k value)
        knn: Scikit-learn KNeighborsClassifier model
        is_trained: Whether the model has been trained

    Example:
        >>> classifier = KNNClassifier(config)
        >>> classifier.train(X_train, y_train)
        >>> predictions = classifier.predict(X_test)
        >>> metrics = classifier.get_metrics(y_test, predictions)
        >>> print(metrics['accuracy'])
        0.933
    """

    def __init__(self, config: Config):
        """
        Initialize KNN classifier with configuration.

        Args:
            config: Configuration object with KNN parameters
        """
        self.n_neighbors = config.knn_neighbors  # Default: 5

        # Initialize KNN with parameters optimized for normalized vectors
        self.knn = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights="distance",  # Weight by inverse distance
            metric="euclidean",  # Standard distance metric
            algorithm="auto",  # Automatically choose best algorithm
            n_jobs=-1,  # Use all CPU cores
        )

        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train KNN classifier on labeled data.

        Args:
            X_train: Training vectors of shape (n_samples, n_features)
            y_train: Training labels of shape (n_samples,)

        Raises:
            ValueError: If X_train and y_train have mismatched lengths

        Example:
            >>> classifier.train(training_vectors, training_labels)
            >>> classifier.is_trained
            True
        """
        # Validate inputs
        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train and y_train length mismatch: "
                f"{len(X_train)} vs {len(y_train)}"
            )

        if len(X_train) < self.n_neighbors:
            raise ValueError(
                f"Need at least {self.n_neighbors} training samples, "
                f"got {len(X_train)}"
            )

        # Fit KNN model
        self.knn.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels for test data.

        Args:
            X_test: Test vectors of shape (n_samples, n_features)

        Returns:
            Predicted labels of shape (n_samples,)

        Raises:
            RuntimeError: If train() hasn't been called yet

        Example:
            >>> predictions = classifier.predict(test_vectors)
            >>> predictions.shape
            (30,)
        """
        if not self.is_trained:
            raise RuntimeError("Must call train() before predict()")

        return self.knn.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for test data.

        Returns probability estimates for each class. Useful for
        understanding confidence of predictions.

        Args:
            X_test: Test vectors of shape (n_samples, n_features)

        Returns:
            Probability array of shape (n_samples, n_classes)

        Raises:
            RuntimeError: If train() hasn't been called yet

        Example:
            >>> proba = classifier.predict_proba(test_vectors)
            >>> proba.shape
            (30, 3)
            >>> proba[0].sum()  # Probabilities sum to 1
            1.0
        """
        if not self.is_trained:
            raise RuntimeError("Must call train() before predict_proba()")

        return self.knn.predict_proba(X_test)

    def get_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        category_names: Optional[list] = None,
    ) -> Dict:
        """
        Calculate classification metrics.

        Computes comprehensive metrics including:
        - Overall accuracy
        - Per-class precision, recall, F1-score
        - Confusion matrix

        Args:
            y_true: True labels
            y_pred: Predicted labels
            category_names: Optional names for categories (for reporting)

        Returns:
            Dictionary with all metrics

        Example:
            >>> metrics = classifier.get_metrics(y_test, predictions,
            ...     category_names=['animals', 'music', 'food'])
            >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
            Accuracy: 93.33%
        """
        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Per-class metrics (precision, recall, F1)
        report = classification_report(
            y_true,
            y_pred,
            target_names=category_names,
            output_dict=True,
            zero_division=0,
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Organize metrics
        metrics = {
            "accuracy": float(accuracy),
            "confusion_matrix": conf_matrix.tolist(),
        }

        # Add per-class metrics if category names provided
        if category_names:
            for category in category_names:
                if category in report:
                    metrics[category] = {
                        "precision": report[category]["precision"],
                        "recall": report[category]["recall"],
                        "f1_score": report[category]["f1-score"],
                        "support": int(report[category]["support"]),
                    }

        # Add macro averages
        if "macro avg" in report:
            metrics["macro_avg"] = {
                "precision": report["macro avg"]["precision"],
                "recall": report["macro avg"]["recall"],
                "f1_score": report["macro avg"]["f1-score"],
            }

        return metrics
