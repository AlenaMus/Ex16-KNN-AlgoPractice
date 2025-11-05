"""
Metrics calculation utilities.

Helper functions for computing and formatting ML metrics.
"""

from typing import Dict, Any
import numpy as np


def calculate_metrics(
    clustering_metrics: Dict, classification_metrics: Dict
) -> Dict[str, Any]:
    """
    Combine and format metrics from clustering and classification.

    Args:
        clustering_metrics: Metrics from K-Means clustering
        classification_metrics: Metrics from KNN classification

    Returns:
        Combined metrics dictionary

    Example:
        >>> combined = calculate_metrics(kmeans_metrics, knn_metrics)
        >>> print(combined['kmeans']['silhouette_score'])
        0.742
    """
    return {"kmeans": clustering_metrics, "knn": classification_metrics}


def format_metrics_summary(metrics: Dict[str, Any]) -> str:
    """
    Format metrics as human-readable summary string.

    Args:
        metrics: Combined metrics dictionary

    Returns:
        Formatted string summary

    Example:
        >>> summary = format_metrics_summary(metrics)
        >>> print(summary)
    """
    lines = []
    lines.append("=" * 50)
    lines.append("METRICS SUMMARY")
    lines.append("=" * 50)

    # K-Means metrics
    if "kmeans" in metrics:
        lines.append("\nK-Means Clustering:")
        km = metrics["kmeans"]
        lines.append(f"  Silhouette Score: {km.get('silhouette_score', 0):.3f}")
        lines.append(f"  Inertia: {km.get('inertia', 0):.2f}")
        if "purity" in km:
            lines.append(f"  Purity: {km['purity']:.3f}")

    # KNN metrics
    if "knn" in metrics:
        lines.append("\nKNN Classification:")
        knn = metrics["knn"]
        lines.append(f"  Accuracy: {knn.get('accuracy', 0):.2%}")

        # Per-category metrics
        for category in ["animals", "music", "food"]:
            if category in knn:
                cat_metrics = knn[category]
                lines.append(f"\n  {category.capitalize()}:")
                lines.append(f"    Precision: {cat_metrics['precision']:.3f}")
                lines.append(f"    Recall: {cat_metrics['recall']:.3f}")
                lines.append(f"    F1-Score: {cat_metrics['f1_score']:.3f}")

    lines.append("\n" + "=" * 50)
    return "\n".join(lines)
