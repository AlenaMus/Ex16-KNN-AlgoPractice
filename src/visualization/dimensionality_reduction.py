"""
Dimensionality reduction for visualization.

Reduces high-dimensional vectors to 2D for scatter plot visualization.
Supports UMAP, t-SNE, and PCA methods.
"""

import numpy as np
from typing import Literal


def reduce_dimensions(
    vectors: np.ndarray,
    method: Literal["umap", "tsne", "pca"] = "umap",
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce high-dimensional vectors to 2D for visualization.

    Transforms vectors from embedding space (1536D) to 2D while
    preserving relative distances and structure as much as possible.

    Args:
        vectors: Array of shape (n_samples, n_features)
        method: Reduction method - 'umap', 'tsne', or 'pca'
        random_state: Random seed for reproducibility

    Returns:
        2D array of shape (n_samples, 2)

    Raises:
        ValueError: If method is unknown or vectors have wrong shape

    Example:
        >>> vectors_2d = reduce_dimensions(vectors, method='umap')
        >>> vectors_2d.shape
        (30, 2)
    """
    # Validate input
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {vectors.shape}")

    if method == "umap":
        return _reduce_umap(vectors, random_state)
    elif method == "tsne":
        return _reduce_tsne(vectors, random_state)
    elif method == "pca":
        return _reduce_pca(vectors, random_state)
    else:
        raise ValueError(f"Unknown reduction method: {method}")


def _reduce_umap(vectors: np.ndarray, random_state: int) -> np.ndarray:
    """Reduce using UMAP (recommended for speed and quality)."""
    from umap import UMAP

    reducer = UMAP(
        n_components=2,
        n_neighbors=15,  # Balance local/global structure
        min_dist=0.1,  # Minimum distance between points
        metric="euclidean",
        random_state=random_state,
    )
    return reducer.fit_transform(vectors)


def _reduce_tsne(vectors: np.ndarray, random_state: int) -> np.ndarray:
    """Reduce using t-SNE (good quality but slower)."""
    from sklearn.manifold import TSNE

    reducer = TSNE(
        n_components=2,
        perplexity=min(30, len(vectors) - 1),  # Adjust for small datasets
        random_state=random_state,
        n_iter=1000,
    )
    return reducer.fit_transform(vectors)


def _reduce_pca(vectors: np.ndarray, random_state: int) -> np.ndarray:
    """Reduce using PCA (fast but linear, may lose structure)."""
    from sklearn.decomposition import PCA

    reducer = PCA(n_components=2, random_state=random_state)
    return reducer.fit_transform(vectors)
