"""
Integration test for full pipeline.

Tests end-to-end execution with small dataset.
"""

import pytest
import numpy as np
from src.core.config import Config
from src.core.utils import normalize_vectors, validate_normalization


def test_normalize_vectors():
    """Test vector normalization."""
    vectors = np.array([[3, 4], [5, 12]])
    normalized = normalize_vectors(vectors)

    # Check norms are 1.0
    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms, 1.0)


def test_validate_normalization_success():
    """Test normalization validation passes for normalized vectors."""
    vectors = np.array([[1, 0], [0, 1]])
    validate_normalization(vectors)  # Should not raise


def test_validate_normalization_failure():
    """Test normalization validation fails for unnormalized vectors."""
    vectors = np.array([[3, 4], [5, 12]])  # Not normalized

    with pytest.raises(ValueError, match="not normalized"):
        validate_normalization(vectors)


def test_config_from_env(tmp_path):
    """Test configuration loading from .env file."""
    # Create temporary .env file
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-test-key-12345")

    config = Config.from_env(str(env_file))
    assert config.openai_api_key == "sk-test-key-12345"
    assert config.openai_model == "text-embedding-3-small"
