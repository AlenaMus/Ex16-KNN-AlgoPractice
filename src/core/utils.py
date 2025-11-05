"""
Utility functions for the classification system.

Provides helper functions for:
- Vector normalization and validation
- File operations and timestamp generation
- Logging setup
- JSON serialization
"""

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict
import numpy as np


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize vectors to unit length using L2 norm.

    Applies L2 normalization so that each vector has magnitude 1:
    v_normalized = v / ||v||₂

    This is critical for KNN with Euclidean distance, as it ensures
    fair comparison between vectors regardless of their original magnitude.

    Args:
        vectors: NumPy array of shape (n_samples, n_features)

    Returns:
        Normalized vectors with same shape, where ||v|| = 1 for each vector

    Raises:
        ValueError: If input is not a 2D array or contains invalid values

    Example:
        >>> vectors = np.array([[3, 4], [5, 12]])
        >>> normalized = normalize_vectors(vectors)
        >>> np.linalg.norm(normalized[0])  # Should be 1.0
        1.0
        >>> np.linalg.norm(normalized[1])  # Should be 1.0
        1.0
    """
    # Validate input shape
    if vectors.ndim != 2:
        raise ValueError(
            f"Expected 2D array, got shape {vectors.shape}. "
            "Input should be (n_samples, n_features)."
        )

    # Check for NaN or Inf values
    if not np.isfinite(vectors).all():
        raise ValueError("Input vectors contain NaN or Inf values")

    # Calculate L2 norm for each vector (row-wise)
    # keepdims=True preserves shape for broadcasting
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    # Avoid division by zero: replace zero norms with 1
    # (zero vectors remain zero after normalization)
    norms = np.where(norms == 0, 1, norms)

    # Divide each vector by its norm
    normalized = vectors / norms

    return normalized


def validate_normalization(vectors: np.ndarray, tolerance: float = 1e-5) -> None:
    """
    Validate that vectors are normalized (||v|| ≈ 1).

    Checks that all vectors have L2 norm approximately equal to 1.0
    within the specified tolerance. This is a critical validation step
    before running KNN classification.

    Args:
        vectors: NumPy array of shape (n_samples, n_features)
        tolerance: Absolute tolerance for norm check (default: 1e-5)

    Raises:
        ValueError: If any vector's norm deviates from 1.0 by more than tolerance

    Example:
        >>> vectors = np.random.rand(10, 100)
        >>> normalized = normalize_vectors(vectors)
        >>> validate_normalization(normalized)  # Should not raise
    """
    # Calculate L2 norm for each vector
    norms = np.linalg.norm(vectors, axis=1)

    # Check if all norms are approximately 1.0
    if not np.allclose(norms, 1.0, atol=tolerance):
        min_norm = norms.min()
        max_norm = norms.max()
        raise ValueError(
            f"Vectors not normalized within tolerance {tolerance}!\n"
            f"Norm range: [{min_norm:.6f}, {max_norm:.6f}]\n"
            f"Expected: all norms ≈ 1.0"
        )


def generate_timestamp() -> str:
    """
    Generate timestamp string for filenames.

    Creates a timestamp in the format YYYYMMDD_HHMMSS for use in
    output filenames. Ensures all files from the same run have the
    same timestamp.

    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS

    Example:
        >>> timestamp = generate_timestamp()
        >>> print(timestamp)
        20251104_143022
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_results_folder(folder_path: str = "results") -> None:
    """
    Create results folder if it doesn't exist.

    Ensures the output directory exists before saving files.
    Creates all intermediate directories if needed.

    Args:
        folder_path: Path to results folder (default: "results")

    Example:
        >>> ensure_results_folder("results")
        >>> os.path.exists("results")
        True
    """
    # exist_ok=True prevents error if folder already exists
    os.makedirs(folder_path, exist_ok=True)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save dictionary to JSON file with proper formatting.

    Serializes data to JSON with indentation for readability.
    Handles NumPy arrays and datetime objects by converting them
    to standard Python types.

    Args:
        data: Dictionary to save
        filepath: Path to output JSON file

    Raises:
        IOError: If file cannot be written
        TypeError: If data contains non-serializable objects

    Example:
        >>> data = {"accuracy": 0.95, "samples": 30}
        >>> save_json(data, "results/metrics.json")
    """

    # Custom JSON encoder to handle NumPy and datetime types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, datetime):
                return obj.isoformat()
            return super().default(obj)

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Write JSON with formatting
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)


def setup_logger(
    name: str, log_file: str = None, level: int = logging.INFO
) -> logging.Logger:
    """
    Set up logger with console and optional file output.

    Creates a logger that outputs to console with colored formatting
    and optionally to a file. Each logger instance is named and can
    have its own log level.

    Args:
        name: Logger name (usually __name__ of calling module)
        log_file: Optional path to log file
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger(__name__, "results/run.log")
        >>> logger.info("Pipeline started")
        2025-11-04 14:30:22 - INFO - Pipeline started
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create console handler with formatted output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Format: timestamp - name - level - message
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if log file specified
    if log_file:
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
