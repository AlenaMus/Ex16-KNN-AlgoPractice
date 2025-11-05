"""
Vectorization Agent for OpenAI embeddings.

This agent handles all interactions with the OpenAI API for converting
sentences to vector embeddings. Includes retry logic, batch processing,
and automatic normalization.
"""

import time
import logging
from typing import List
import numpy as np
from openai import OpenAI
import openai

from src.core.config import Config
from src.core.utils import normalize_vectors, validate_normalization


class VectorizationAgent:
    """
    Agent for converting sentences to vector embeddings using OpenAI API.

    This agent encapsulates all OpenAI API interactions, handling:
    - Batch processing for efficiency
    - Automatic retry with exponential backoff
    - Vector normalization and validation
    - Progress tracking with tqdm
    - Comprehensive error handling

    Attributes:
        client: OpenAI API client instance
        model: Name of embedding model to use
        embedding_dim: Dimension of embedding vectors (1536 for text-embedding-3-small)
        batch_size: Maximum sentences per API call
        max_retries: Maximum retry attempts for failed calls
        logger: Logger instance for tracking operations

    Example:
        >>> config = Config.from_env()
        >>> agent = VectorizationAgent(config)
        >>> sentences = ["The cat sits", "Dogs bark"]
        >>> vectors = agent.vectorize(sentences)
        >>> vectors.shape
        (2, 1536)
    """

    def __init__(self, config: Config):
        """
        Initialize vectorization agent with configuration.

        Args:
            config: Configuration object with OpenAI settings
        """
        # Initialize OpenAI client with API key
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = config.openai_model
        self.embedding_dim = 1536  # Fixed for text-embedding-3-small
        self.batch_size = config.batch_size
        self.max_retries = config.max_retries

        # Set up logging for this agent
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Initialized VectorizationAgent with model: {self.model}")

    def vectorize(
        self, sentences: List[str], normalize: bool = True, show_progress: bool = True
    ) -> np.ndarray:
        """
        Convert list of sentences to normalized vector embeddings.

        This is the main method for vectorization. It handles batching,
        API calls, normalization, and validation automatically.

        Args:
            sentences: List of text strings to embed
            normalize: Whether to apply L2 normalization (default: True)
            show_progress: Whether to show progress bar (default: True)

        Returns:
            NumPy array of shape (n_sentences, 1536) with normalized vectors

        Raises:
            ValueError: If sentences list is empty
            openai.APIError: If API calls fail after all retries

        Example:
            >>> agent = VectorizationAgent(config)
            >>> vectors = agent.vectorize(["Hello world", "Good morning"])
            >>> np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
            True
        """
        # Validate input
        if not sentences:
            raise ValueError("Sentences list cannot be empty")

        self.logger.info(f"Vectorizing {len(sentences)} sentences...")

        # Process in batches if needed
        if len(sentences) <= self.batch_size:
            # Small enough for single call
            embeddings = self._call_api_with_retry(sentences)
        else:
            # Need to batch
            embeddings = self._vectorize_batch(sentences, show_progress)

        # Apply L2 normalization if requested
        if normalize:
            self.logger.debug("Applying L2 normalization...")
            embeddings = normalize_vectors(embeddings)

            # Validate normalization succeeded
            validate_normalization(embeddings, tolerance=1e-5)
            self.logger.debug("Normalization validated successfully")

        self.logger.info(f"Vectorization complete. Shape: {embeddings.shape}")
        return embeddings

    def _vectorize_batch(self, sentences: List[str], show_progress: bool) -> np.ndarray:
        """
        Process large sentence lists in batches with progress tracking.

        Splits sentences into batches and processes each batch separately.
        Shows progress bar if requested using tqdm.

        Args:
            sentences: List of sentences to vectorize
            show_progress: Whether to display progress bar

        Returns:
            Concatenated embeddings from all batches
        """
        from tqdm import tqdm

        all_embeddings = []

        # Create batches
        batches = [
            sentences[i : i + self.batch_size]
            for i in range(0, len(sentences), self.batch_size)
        ]

        # Process each batch with progress bar
        iterator = tqdm(batches, desc="Vectorizing") if show_progress else batches

        for batch in iterator:
            embeddings = self._call_api_with_retry(batch)
            all_embeddings.append(embeddings)

        # Concatenate all batches vertically
        return np.vstack(all_embeddings)

    def _call_api_with_retry(
        self, sentences: List[str], attempt: int = 1
    ) -> np.ndarray:
        """
        Call OpenAI API with exponential backoff retry logic.

        Implements robust error handling for various API failures:
        - Rate limit errors: Retry with exponential backoff
        - Connection errors: Retry with backoff
        - Authentication errors: Raise immediately (no retry)
        - Other API errors: Retry with backoff

        Args:
            sentences: List of sentences for this batch
            attempt: Current attempt number (used for recursion)

        Returns:
            NumPy array of embeddings for this batch

        Raises:
            openai.AuthenticationError: If API key is invalid
            openai.APIError: If all retry attempts fail
        """
        try:
            # Call OpenAI embeddings API
            response = self.client.embeddings.create(
                model=self.model, input=sentences, encoding_format="float"
            )

            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]

            # Convert to NumPy array with float32 for efficiency
            return np.array(embeddings, dtype=np.float32)

        except openai.RateLimitError as e:
            # Rate limit hit - retry with exponential backoff
            if attempt <= self.max_retries:
                delay = 2 ** (attempt - 1)  # 1s, 2s, 4s
                self.logger.warning(
                    f"Rate limited. Retrying in {delay}s... "
                    f"(attempt {attempt}/{self.max_retries})"
                )
                time.sleep(delay)
                return self._call_api_with_retry(sentences, attempt + 1)
            else:
                self.logger.error("Max retries exceeded for rate limit")
                raise

        except openai.AuthenticationError as e:
            # Invalid API key - don't retry
            self.logger.error("Authentication failed. Check your API key in .env file")
            raise

        except openai.APIConnectionError as e:
            # Network error - retry with backoff
            if attempt <= self.max_retries:
                delay = 2 ** (attempt - 1)
                self.logger.warning(
                    f"Connection error. Retrying in {delay}s... "
                    f"(attempt {attempt}/{self.max_retries})"
                )
                time.sleep(delay)
                return self._call_api_with_retry(sentences, attempt + 1)
            else:
                self.logger.error("Max retries exceeded for connection error")
                raise

        except openai.APIError as e:
            # General API error - retry with backoff
            if attempt <= self.max_retries:
                delay = 2 ** (attempt - 1)
                self.logger.warning(
                    f"API error: {str(e)}. Retrying in {delay}s... "
                    f"(attempt {attempt}/{self.max_retries})"
                )
                time.sleep(delay)
                return self._call_api_with_retry(sentences, attempt + 1)
            else:
                self.logger.error(f"Max retries exceeded. Last error: {str(e)}")
                raise
