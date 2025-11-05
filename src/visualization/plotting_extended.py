"""
Extended plotting functions for before/after visualizations.

Creates side-by-side comparison plots showing data before and after
applying ML algorithms, with comprehensive metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import Dict, Optional
from src.core.config import Config


def create_kmeans_before_after_plot(
    vectors_2d: np.ndarray,
    true_labels: np.ndarray,
    cluster_labels: np.ndarray,
    config: Config,
    metrics: Dict,
    cluster_centers_2d: Optional[np.ndarray] = None,
    output_path: str = "results/kmeans_before_after.png",
) -> None:
    """
    Create before/after visualization for K-Means clustering.

    Shows original labeled data (left) and clustered data (right) side by side.

    Args:
        vectors_2d: 2D coordinates from dimensionality reduction
        true_labels: Original category labels (0=animals, 1=music, 2=food)
        cluster_labels: K-Means cluster assignments
        config: Configuration object
        metrics: Dictionary with clustering metrics
        cluster_centers_2d: Optional 2D cluster centers
        output_path: Path to save PNG file
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=300)

    # LEFT PLOT: Before K-Means (original labels)
    for idx, category in enumerate(config.categories):
        mask = true_labels == idx
        color = config.colors[category]

        ax1.scatter(
            vectors_2d[mask, 0],
            vectors_2d[mask, 1],
            c=color,
            label=f"{category.capitalize()} - {mask.sum()} samples",
            s=100,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

    ax1.set_title("BEFORE K-Means: Original Categories", fontsize=16, fontweight="bold", pad=20)
    ax1.set_xlabel(f"{config.reduction_method.upper()} Dimension 1", fontsize=12)
    ax1.set_ylabel(f"{config.reduction_method.upper()} Dimension 2", fontsize=12)
    ax1.legend(loc="best", fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # RIGHT PLOT: After K-Means (cluster assignments)
    # Color by original category but grouped by clusters
    for idx, category in enumerate(config.categories):
        mask = true_labels == idx
        color = config.colors[category]

        ax2.scatter(
            vectors_2d[mask, 0],
            vectors_2d[mask, 1],
            c=color,
            label=f"{category.capitalize()}",
            s=100,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

    # Add cluster centers
    if cluster_centers_2d is not None:
        ax2.scatter(
            cluster_centers_2d[:, 0],
            cluster_centers_2d[:, 1],
            marker="X",
            s=400,
            c="black",
            edgecolors="white",
            linewidth=3,
            label="Cluster Centers",
            zorder=10,
        )

    ax2.set_title("AFTER K-Means: Clustered Data (k=3)", fontsize=16, fontweight="bold", pad=20)
    ax2.set_xlabel(f"{config.reduction_method.upper()} Dimension 1", fontsize=12)
    ax2.set_ylabel(f"{config.reduction_method.upper()} Dimension 2", fontsize=12)
    ax2.legend(loc="best", fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle="--")

    # Add comprehensive metrics box
    metrics_text = (
        f"K-Means Clustering Metrics:\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Clusters: {config.kmeans_clusters}\n"
        f"Samples: {len(vectors_2d)}\n"
        f"Silhouette: {metrics.get('silhouette_score', 0):.3f}\n"
        f"Purity: {metrics.get('purity', 0):.3f}\n"
        f"ARI: {metrics.get('ari', 0):.3f}\n"
        f"Inertia: {metrics.get('inertia', 0):.2f}\n"
        f"Reduction: {config.reduction_method.upper()}\n"
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    ax2.text(
        0.02,
        0.98,
        metrics_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
        family="monospace",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_knn_before_after_plot(
    train_2d: np.ndarray,
    test_2d: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    predictions: np.ndarray,
    config: Config,
    metrics: Dict,
    output_path: str = "results/knn_before_after.png",
) -> None:
    """
    Create before/after visualization for KNN classification.

    Shows test data before classification (left) and after (right) with predictions.

    Args:
        train_2d: 2D training data coordinates
        test_2d: 2D test data coordinates
        train_labels: Training labels
        test_labels: True test labels
        predictions: KNN predictions for test data
        config: Configuration object
        metrics: Dictionary with classification metrics
        output_path: Path to save PNG file
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=300)

    # LEFT PLOT: Before KNN (training data + unlabeled test data)
    # Show training data in color
    for idx, category in enumerate(config.categories):
        mask = train_labels == idx
        color = config.colors[category]

        ax1.scatter(
            train_2d[mask, 0],
            train_2d[mask, 1],
            c=color,
            label=f"Train: {category.capitalize()}",
            s=80,
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
        )

    # Show test data as gray (unlabeled)
    ax1.scatter(
        test_2d[:, 0],
        test_2d[:, 1],
        c="lightgray",
        label="Test (Unlabeled)",
        s=100,
        alpha=0.8,
        edgecolors="black",
        linewidth=1.5,
        marker="s",
    )

    ax1.set_title(
        "BEFORE KNN: Training Data + Unlabeled Test", fontsize=16, fontweight="bold", pad=20
    )
    ax1.set_xlabel(f"{config.reduction_method.upper()} Dimension 1", fontsize=12)
    ax1.set_ylabel(f"{config.reduction_method.upper()} Dimension 2", fontsize=12)
    ax1.legend(loc="best", fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # RIGHT PLOT: After KNN (classified test data)
    # Show training data faded
    for idx, category in enumerate(config.categories):
        mask = train_labels == idx
        color = config.colors[category]

        ax2.scatter(
            train_2d[mask, 0],
            train_2d[mask, 1],
            c=color,
            s=40,
            alpha=0.2,
            edgecolors="none",
        )

    # Show test data colored by predictions
    for idx, category in enumerate(config.categories):
        mask = predictions == idx
        color = config.colors[category]
        category_accuracy = _get_category_accuracy(test_labels, predictions, idx)

        ax2.scatter(
            test_2d[mask, 0],
            test_2d[mask, 1],
            c=color,
            label=f"{category.capitalize()} - Acc: {category_accuracy:.1%}",
            s=100,
            alpha=0.8,
            edgecolors="black",
            linewidth=0.5,
            marker="s",
        )

    # Highlight misclassifications
    misclassified = test_labels != predictions
    if misclassified.any():
        ax2.scatter(
            test_2d[misclassified, 0],
            test_2d[misclassified, 1],
            facecolors="none",
            edgecolors="red",
            s=200,
            linewidth=3,
            label=f"Misclassified - {misclassified.sum()}",
        )

    ax2.set_title(
        f"AFTER KNN: Classified Test Data (k={config.knn_neighbors})",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax2.set_xlabel(f"{config.reduction_method.upper()} Dimension 1", fontsize=12)
    ax2.set_ylabel(f"{config.reduction_method.upper()} Dimension 2", fontsize=12)
    ax2.legend(loc="best", fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle="--")

    # Add comprehensive metrics box
    metrics_text = (
        f"KNN Classification Metrics:\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"K Value: {config.knn_neighbors}\n"
        f"Train: {len(train_2d)} Test: {len(test_2d)}\n"
        f"Accuracy: {metrics.get('accuracy', 0):.2%}\n"
        f"Precision: {metrics.get('macro_avg', {}).get('precision', 0):.3f}\n"
        f"Recall: {metrics.get('macro_avg', {}).get('recall', 0):.3f}\n"
        f"F1-Score: {metrics.get('macro_avg', {}).get('f1_score', 0):.3f}\n"
        f"Reduction: {config.reduction_method.upper()}\n"
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    ax2.text(
        0.02,
        0.98,
        metrics_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.9),
        family="monospace",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _get_category_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, category_idx: int
) -> float:
    """Calculate accuracy for a specific category."""
    mask = y_true == category_idx
    if mask.sum() == 0:
        return 0.0
    correct = (y_pred[mask] == category_idx).sum()
    return correct / mask.sum()
