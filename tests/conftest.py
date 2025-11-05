"""
Pytest configuration and shared fixtures.

Provides common test fixtures for sentence data, vectors, and configuration.
"""

import pytest
import numpy as np
from src.core.config import Config


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return Config(
        openai_api_key="sk-test-key",
        sentences_per_category=5,
        knn_neighbors=3,
        random_state=42,
    )


@pytest.fixture
def sample_sentences():
    """Create sample sentences for testing."""
    return {
        "animals": ["The cat sits on the mat", "Dogs bark at strangers"],
        "music": ["The piano plays softly", "Jazz improvisation requires skill"],
        "food": ["Fresh bread smells wonderful", "Chocolate tastes sweet"],
    }


@pytest.fixture
def sample_vectors():
    """Create sample normalized vectors for testing."""
    np.random.seed(42)
    vectors = np.random.rand(10, 128)  # 10 samples, 128 features
    # Normalize to unit length
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


@pytest.fixture
def sample_labels():
    """Create sample labels for testing."""
    return np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
