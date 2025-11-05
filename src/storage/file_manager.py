"""
File management for saving results.

Handles saving of all output files including visualizations, data,
vectors, metrics, and logs.
"""

import os
import numpy as np
from typing import Dict, List
from src.core.utils import save_json, generate_timestamp, ensure_results_folder


class FileManager:
    """
    Manage saving of all pipeline results.

    Provides centralized file management with consistent naming
    and organization. All files use the same timestamp for easy
    identification of runs.

    Attributes:
        results_folder: Path to results directory
        timestamp: Timestamp string for this run

    Example:
        >>> manager = FileManager("results")
        >>> manager.save_training_data(train_sentences)
        >>> manager.save_vectors(train_vectors, test_vectors, ...)
    """

    def __init__(self, results_folder: str = "results", sentences_per_category: int = 10):
        """
        Initialize file manager.

        Args:
            results_folder: Path to results directory
            sentences_per_category: Number of sentences per category (for subfolder naming)
        """
        self.base_results_folder = results_folder
        self.sentences_per_category = sentences_per_category
        self.timestamp = generate_timestamp()

        # Create subfolder based on sentence count
        self.results_folder = os.path.join(
            self.base_results_folder,
            str(sentences_per_category)
        )

        # Ensure results folder exists
        ensure_results_folder(self.results_folder)

    def save_training_data(self, sentences: Dict[str, List[str]]) -> str:
        """
        Save training sentences to JSON file.

        Args:
            sentences: Dictionary with category keys and sentence lists

        Returns:
            Path to saved file
        """
        filename = f"training_data_{self.timestamp}.json"
        filepath = os.path.join(self.results_folder, filename)

        data = {
            **sentences,
            "metadata": {
                "timestamp": self.timestamp,
                "total_sentences": sum(len(v) for v in sentences.values()),
                "categories": list(sentences.keys()),
            },
        }

        save_json(data, filepath)
        return filepath

    def save_test_data(self, sentences: Dict[str, List[str]]) -> str:
        """Save test sentences to JSON file."""
        filename = f"test_data_{self.timestamp}.json"
        filepath = os.path.join(self.results_folder, filename)

        data = {
            **sentences,
            "metadata": {
                "timestamp": self.timestamp,
                "total_sentences": sum(len(v) for v in sentences.values()),
                "categories": list(sentences.keys()),
            },
        }

        save_json(data, filepath)
        return filepath

    def save_vectors(
        self,
        training_vectors: np.ndarray,
        test_vectors: np.ndarray,
        training_labels: np.ndarray,
        test_labels: np.ndarray,
    ) -> str:
        """
        Save vector embeddings to NPZ file.

        Args:
            training_vectors: Training embeddings
            test_vectors: Test embeddings
            training_labels: Training category labels
            test_labels: Test category labels

        Returns:
            Path to saved file
        """
        filename = f"vectors_{self.timestamp}.npz"
        filepath = os.path.join(self.results_folder, filename)

        np.savez(
            filepath,
            training_vectors=training_vectors,
            test_vectors=test_vectors,
            training_labels=training_labels,
            test_labels=test_labels,
            normalization_applied=True,
        )

        return filepath

    def save_metrics(self, metrics: Dict) -> str:
        """
        Save all metrics to JSON file.

        Args:
            metrics: Combined metrics dictionary

        Returns:
            Path to saved file
        """
        filename = f"metrics_{self.timestamp}.json"
        filepath = os.path.join(self.results_folder, filename)

        # Add timestamp to metrics
        metrics["timestamp"] = self.timestamp

        save_json(metrics, filepath)
        return filepath

    def get_kmeans_plot_path(self) -> str:
        """Get path for K-Means visualization."""
        return os.path.join(
            self.results_folder, f"kmeans_clustering_{self.timestamp}.png"
        )

    def get_kmeans_before_after_path(self) -> str:
        """Get path for K-Means before/after visualization."""
        return os.path.join(
            self.results_folder, f"kmeans_before_after_{self.timestamp}.png"
        )

    def get_knn_plot_path(self) -> str:
        """Get path for KNN visualization."""
        return os.path.join(
            self.results_folder, f"knn_classification_{self.timestamp}.png"
        )

    def get_knn_before_after_path(self) -> str:
        """Get path for KNN before/after visualization."""
        return os.path.join(
            self.results_folder, f"knn_before_after_{self.timestamp}.png"
        )

    def get_log_path(self) -> str:
        """Get path for execution log."""
        return os.path.join(self.results_folder, f"run_log_{self.timestamp}.txt")
