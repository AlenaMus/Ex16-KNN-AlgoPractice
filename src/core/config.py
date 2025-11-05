"""
Configuration management for K-Means & KNN classification system.

This module handles loading configuration from environment variables and
provides default values for all system parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple
from dotenv import load_dotenv


@dataclass
class Config:
    """
    Configuration class for the classification pipeline.

    Loads settings from .env file and provides defaults for all parameters.
    All configuration is immutable after initialization for consistency.

    Attributes:
        openai_api_key: OpenAI API key for embeddings
        openai_model: Model name for embeddings (default: text-embedding-3-small)
        openai_timeout: API timeout in seconds (default: 30)
        sentences_per_category: Number of sentences to generate per category
        categories: Tuple of category names (animals, music, food)
        kmeans_clusters: Number of clusters for K-Means (fixed at 3)
        knn_neighbors: K value for KNN algorithm (default: 5)
        random_state: Random seed for reproducibility (default: 42)
        colors: Dictionary mapping categories to hex colors
        reduction_method: Dimensionality reduction method (umap/tsne/pca)
        results_folder: Path to results output folder
        batch_size: Batch size for API calls (default: 100)
        max_retries: Maximum API retry attempts (default: 3)

    Example:
        >>> config = Config.from_env()
        >>> print(config.openai_model)
        text-embedding-3-small
        >>> print(config.colors['animals'])
        #00FF00
    """

    # OpenAI API settings
    openai_api_key: str
    openai_model: str = "text-embedding-3-small"
    openai_timeout: int = 30

    # Data generation settings
    sentences_per_category: int = 10
    categories: Tuple[str, str, str] = ("animals", "music", "food")

    # Machine learning parameters
    kmeans_clusters: int = 3  # Fixed: matches number of categories
    knn_neighbors: int = 5
    random_state: int = 42

    # Visualization settings
    colors: Dict[str, str] = field(default_factory=dict)
    reduction_method: str = "umap"

    # Output settings
    results_folder: str = "results"

    # API settings
    batch_size: int = 100  # Max sentences per API call
    max_retries: int = 3  # API retry attempts

    def __post_init__(self):
        """
        Initialize default color mapping after instance creation.

        Sets the standard color scheme:
        - Animals: Green (#00FF00)
        - Music: Blue (#0000FF)
        - Food: Red (#FF0000)
        """
        if not self.colors:
            self.colors = {
                "animals": "#00FF00",  # Green
                "music": "#0000FF",  # Blue
                "food": "#FF0000",  # Red
            }

        # Validate that categories match color keys
        if set(self.categories) != set(self.colors.keys()):
            raise ValueError(
                f"Categories {self.categories} don't match color keys {self.colors.keys()}"
            )

    @classmethod
    def from_env(cls, env_path: str = ".env") -> "Config":
        """
        Load configuration from environment variables.

        Reads the .env file and creates a Config instance with values from
        environment variables, falling back to defaults where not specified.

        Args:
            env_path: Path to .env file (default: ".env" in current directory)

        Returns:
            Config instance with loaded settings

        Raises:
            ValueError: If OPENAI_API_KEY is not found in environment
            FileNotFoundError: If .env file doesn't exist

        Example:
            >>> config = Config.from_env()
            >>> print(f"Using model: {config.openai_model}")
            Using model: text-embedding-3-small
        """
        # Load environment variables from .env file
        if not os.path.exists(env_path):
            raise FileNotFoundError(
                f".env file not found at {env_path}. "
                "Please create it with your OPENAI_API_KEY."
            )

        load_dotenv(env_path)

        # Get API key (required)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment. "
                "Please set it in your .env file."
            )

        # Validate API key format
        if not api_key.startswith("sk-"):
            raise ValueError(
                f"Invalid OpenAI API key format. Key should start with 'sk-', "
                f"got: {api_key[:10]}..."
            )

        # Create config with environment variables or defaults
        return cls(
            openai_api_key=api_key,
            openai_model=os.getenv("OPENAI_MODEL", "text-embedding-3-small"),
            openai_timeout=int(os.getenv("OPENAI_TIMEOUT", "30")),
        )

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Checks that all parameters are within acceptable ranges and
        have valid values.

        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate numeric parameters are positive
        if self.sentences_per_category <= 0:
            raise ValueError("sentences_per_category must be positive")

        if self.kmeans_clusters != 3:
            raise ValueError("kmeans_clusters must be 3 (fixed for 3 categories)")

        if self.knn_neighbors <= 0:
            raise ValueError("knn_neighbors must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        # Validate reduction method
        valid_methods = ["umap", "tsne", "pca"]
        if self.reduction_method not in valid_methods:
            raise ValueError(
                f"reduction_method must be one of {valid_methods}, "
                f"got: {self.reduction_method}"
            )

        # Validate API key exists
        if not self.openai_api_key:
            raise ValueError("openai_api_key cannot be empty")

    def __repr__(self) -> str:
        """
        String representation of config (with masked API key).

        Returns:
            String representation with sensitive data masked
        """
        masked_key = f"{self.openai_api_key[:10]}...{self.openai_api_key[-4:]}"
        return (
            f"Config(model={self.openai_model}, "
            f"sentences={self.sentences_per_category}, "
            f"k_neighbors={self.knn_neighbors}, "
            f"api_key={masked_key})"
        )
