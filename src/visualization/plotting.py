"""
Plotting functions for K-Means and KNN visualizations.

Creates publication-quality scatter plots with metadata boxes,
legends, and color-coded categories.
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import Dict, Optional
from src.core.config import Config


def create_kmeans_plot(
    vectors_2d: np.ndarray,
    labels: np.ndarray,
    config: Config,
    metrics: Dict,
    cluster_centers_2d: Optional[np.ndarray] = None,
    output_path: str = "results/kmeans_clustering.png",
) -> None:
    """
    Create K-Means clustering visualization.

    Generates scatter plot with:
    - Color-coded points by original category
    - Cluster centers marked with X
    - Legend with category counts
    - Metadata box with algorithm info and metrics

    Args:
        vectors_2d: 2D coordinates from dimensionality reduction
        labels: Original category labels (0=animals, 1=music, 2=food)
        config: Configuration object
        metrics: Dictionary with clustering metrics
        cluster_centers_2d: Optional 2D cluster centers
        output_path: Path to save PNG file
    """
    # Create figure with high resolution
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # Plot each category with its assigned color
    for idx, category in enumerate(config.categories):
        mask = labels == idx
        color = config.colors[category]

        ax.scatter(
            vectors_2d[mask, 0],
            vectors_2d[mask, 1],
            c=color,
            label=f"{category.capitalize()} - {mask.sum()} samples",
            s=100,  # Point size
            alpha=0.7,  # Transparency
            edgecolors="black",
            linewidth=0.5,
        )

    # Plot cluster centers if provided
    if cluster_centers_2d is not None:
        ax.scatter(
            cluster_centers_2d[:, 0],
            cluster_centers_2d[:, 1],
            marker="X",
            s=300,
            c="black",
            edgecolors="white",
            linewidth=2,
            label="Cluster Centers",
            zorder=10,  # Draw on top
        )

    # Set title and labels
    ax.set_title(
        f"K-Means Clustering Results (k={config.kmeans_clusters})",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel(f"{config.reduction_method.upper()} Dimension 1", fontsize=12)
    ax.set_ylabel(f"{config.reduction_method.upper()} Dimension 2", fontsize=12)

    # Add legend
    ax.legend(loc="best", fontsize=10, framealpha=0.9)

    # Add grid for readability
    ax.grid(True, alpha=0.3, linestyle="--")

    # Create metadata box
    metadata_text = _format_kmeans_metadata(metrics, config)
    ax.text(
        0.02,
        0.98,
        metadata_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        family="monospace",
    )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_knn_plot(
    vectors_2d: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: Config,
    metrics: Dict,
    output_path: str = "results/knn_classification.png",
) -> None:
    """
    Create KNN classification visualization.

    Generates scatter plot with:
    - Color-coded points by true category
    - Optional highlighting of misclassifications
    - Legend with per-category accuracy
    - Metadata box with classification metrics

    Args:
        vectors_2d: 2D coordinates from dimensionality reduction
        y_true: True category labels
        y_pred: Predicted category labels
        config: Configuration object
        metrics: Dictionary with classification metrics
        output_path: Path to save PNG file
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # Plot each category
    for idx, category in enumerate(config.categories):
        mask = y_true == idx
        color = config.colors[category]

        # Get accuracy for this category
        category_accuracy = _get_category_accuracy(y_true, y_pred, idx)

        ax.scatter(
            vectors_2d[mask, 0],
            vectors_2d[mask, 1],
            c=color,
            label=f"{category.capitalize()} - {mask.sum()} samples (Acc: {category_accuracy:.1%})",
            s=100,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

    # Highlight misclassifications with red edge
    misclassified = y_true != y_pred
    if misclassified.any():
        ax.scatter(
            vectors_2d[misclassified, 0],
            vectors_2d[misclassified, 1],
            facecolors="none",
            edgecolors="red",
            s=150,
            linewidth=2,
            label=f"Misclassified - {misclassified.sum()} samples",
        )

    # Set title and labels
    ax.set_title(
        f"KNN Classification Results (k={config.knn_neighbors})",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel(f"{config.reduction_method.upper()} Dimension 1", fontsize=12)
    ax.set_ylabel(f"{config.reduction_method.upper()} Dimension 2", fontsize=12)

    # Add legend
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")

    # Create metadata box
    metadata_text = _format_knn_metadata(metrics, config, len(y_true))
    ax.text(
        0.02,
        0.98,
        metadata_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        family="monospace",
    )

    # Save and close
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _format_kmeans_metadata(metrics: Dict, config: Config) -> str:
    """Format metadata text for K-Means plot."""
    return (
        f"Algorithm: K-Means\n"
        f"Clusters: {config.kmeans_clusters}\n"
        f"Total Samples: {sum(metrics.get('cluster_sizes', [0]))}\n"
        f"Silhouette: {metrics.get('silhouette_score', 0):.3f}\n"
        f"Purity: {metrics.get('purity', 0):.3f}\n"
        f"ARI: {metrics.get('ari', 0):.3f}\n"
        f"Inertia: {metrics.get('inertia', 0):.2f}\n"
        f"Reduction: {config.reduction_method.upper()}\n"
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )


def _format_knn_metadata(metrics: Dict, config: Config, n_test: int) -> str:
    """Format metadata text for KNN plot."""
    return (
        f"Algorithm: K-Nearest Neighbors\n"
        f"K Value: {config.knn_neighbors}\n"
        f"Test Samples: {n_test}\n"
        f"Accuracy: {metrics.get('accuracy', 0):.2%}\n"
        f"Precision (avg): {metrics.get('macro_avg', {}).get('precision', 0):.3f}\n"
        f"Recall (avg): {metrics.get('macro_avg', {}).get('recall', 0):.3f}\n"
        f"Reduction: {config.reduction_method.upper()}\n"
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )


def _get_category_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, category_idx: int
) -> float:
    """Calculate accuracy for a specific category."""
    mask = y_true == category_idx
    if mask.sum() == 0:
        return 0.0
    correct = (y_pred[mask] == category_idx).sum()
    return correct / mask.sum()
