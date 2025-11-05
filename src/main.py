"""
Main pipeline orchestration for K-Means & KNN classification system.

Coordinates all components: sentence generation, vectorization,
clustering, classification, visualization, and results storage.
"""

import argparse
import sys
import numpy as np

from src.core.config import Config
from src.core.utils import setup_logger
from src.data.sentence_generator import SentenceGenerator
from agents.vectorization_agent import VectorizationAgent
from src.ml.clustering import KMeansClustering
from src.ml.classification import KNNClassifier
from src.ml.metrics import calculate_metrics, format_metrics_summary
from src.visualization.dimensionality_reduction import reduce_dimensions
from src.visualization.plotting import create_kmeans_plot, create_knn_plot
from src.visualization.plotting_extended import (
    create_kmeans_before_after_plot,
    create_knn_before_after_plot,
)
from src.storage.file_manager import FileManager


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="K-Means & KNN Sentence Classification Pipeline"
    )

    parser.add_argument(
        "--sentences",
        "-s",
        type=int,
        default=10,
        help="Number of sentences per category (default: 10)",
    )

    parser.add_argument(
        "--k-neighbors",
        "-k",
        type=int,
        default=5,
        help="K value for KNN (default: 5)",
    )

    parser.add_argument(
        "--reduction",
        "-r",
        choices=["umap", "tsne", "pca"],
        default="umap",
        help="Dimensionality reduction method (default: umap)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args()


def main():
    """Main pipeline execution."""
    # Parse arguments
    args = parse_arguments()

    # Load configuration
    try:
        config = Config.from_env()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Please check your .env file and ensure OPENAI_API_KEY is set.")
        sys.exit(1)

    # Override config with command-line arguments
    config.sentences_per_category = args.sentences
    config.knn_neighbors = args.k_neighbors
    config.reduction_method = args.reduction

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    # Initialize file manager with sentence count for subfolder organization
    file_manager = FileManager(config.results_folder, config.sentences_per_category)

    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger(__name__, file_manager.get_log_path(), log_level)

    logger.info("=" * 60)
    logger.info("K-MEANS & KNN SENTENCE CLASSIFICATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Configuration: {config}")
    logger.info(f"Sentences per category: {config.sentences_per_category}")

    try:
        # Step 1: Generate sentences
        logger.info("\n[Step 1/14] Generating training sentences...")
        generator = SentenceGenerator(config)
        train_sentences = generator.generate_training_set(config.sentences_per_category)
        logger.info(
            f"Generated {sum(len(v) for v in train_sentences.values())} training sentences"
        )

        logger.info("\n[Step 2/12] Generating test sentences...")
        test_sentences = generator.generate_test_set(config.sentences_per_category)
        logger.info(
            f"Generated {sum(len(v) for v in test_sentences.values())} test sentences"
        )

        # Step 2: Vectorize sentences
        logger.info("\n[Step 3/12] Initializing vectorization agent...")
        agent = VectorizationAgent(config)

        logger.info("\n[Step 4/12] Vectorizing training sentences...")
        train_sentences_flat = [
            s for sentences in train_sentences.values() for s in sentences
        ]
        train_labels = np.array(
            [i for i, cat in enumerate(config.categories) for _ in train_sentences[cat]]
        )
        train_vectors = agent.vectorize(train_sentences_flat, normalize=True)
        logger.info(f"Training vectors shape: {train_vectors.shape}")

        logger.info("\n[Step 5/12] Vectorizing test sentences...")
        test_sentences_flat = [
            s for sentences in test_sentences.values() for s in sentences
        ]
        test_labels = np.array(
            [i for i, cat in enumerate(config.categories) for _ in test_sentences[cat]]
        )
        test_vectors = agent.vectorize(test_sentences_flat, normalize=True)
        logger.info(f"Test vectors shape: {test_vectors.shape}")

        # Step 3: K-Means clustering
        logger.info("\n[Step 6/12] Running K-Means clustering...")
        clustering = KMeansClustering(config)
        cluster_labels = clustering.fit(train_vectors)
        kmeans_metrics = clustering.get_metrics(train_vectors, train_labels)
        logger.info(f"Silhouette score: {kmeans_metrics['silhouette_score']:.3f}")

        # Step 4: Visualize K-Means
        logger.info("\n[Step 7/12] Creating K-Means visualization...")
        train_2d = reduce_dimensions(
            train_vectors, config.reduction_method, config.random_state
        )
        # Use PCA for cluster centers (only 3 points, UMAP may fail on small data)
        centers_2d = reduce_dimensions(
            clustering.get_cluster_centers(),
            "pca",
            config.random_state,
        )
        create_kmeans_plot(
            train_2d,
            train_labels,
            config,
            kmeans_metrics,
            centers_2d,
            file_manager.get_kmeans_plot_path(),
        )
        logger.info(f"Saved K-Means plot: {file_manager.get_kmeans_plot_path()}")

        # Create K-Means before/after comparison
        logger.info("Creating K-Means before/after visualization...")
        create_kmeans_before_after_plot(
            train_2d,
            train_labels,
            cluster_labels,
            config,
            kmeans_metrics,
            centers_2d,
            file_manager.get_kmeans_before_after_path(),
        )
        logger.info(
            f"Saved K-Means before/after plot: {file_manager.get_kmeans_before_after_path()}"
        )

        # Step 5: Train KNN
        logger.info("\n[Step 8/12] Training KNN classifier...")
        classifier = KNNClassifier(config)
        classifier.train(train_vectors, train_labels)
        logger.info("KNN training complete")

        # Step 6: Predict test data
        logger.info("\n[Step 9/12] Predicting test data...")
        predictions = classifier.predict(test_vectors)
        knn_metrics = classifier.get_metrics(
            test_labels, predictions, list(config.categories)
        )
        logger.info(f"Test accuracy: {knn_metrics['accuracy']:.2%}")

        # Step 7: Visualize KNN
        logger.info("\n[Step 10/12] Creating KNN visualization...")
        test_2d = reduce_dimensions(
            test_vectors, config.reduction_method, config.random_state
        )
        create_knn_plot(
            test_2d,
            test_labels,
            predictions,
            config,
            knn_metrics,
            file_manager.get_knn_plot_path(),
        )
        logger.info(f"Saved KNN plot: {file_manager.get_knn_plot_path()}")

        # Create KNN before/after comparison
        logger.info("Creating KNN before/after visualization...")
        create_knn_before_after_plot(
            train_2d,
            test_2d,
            train_labels,
            test_labels,
            predictions,
            config,
            knn_metrics,
            file_manager.get_knn_before_after_path(),
        )
        logger.info(
            f"Saved KNN before/after plot: {file_manager.get_knn_before_after_path()}"
        )

        # Step 8: Save all results
        logger.info("\n[Step 11/12] Saving results...")
        file_manager.save_training_data(train_sentences)
        file_manager.save_test_data(test_sentences)
        file_manager.save_vectors(
            train_vectors, test_vectors, train_labels, test_labels
        )

        combined_metrics = calculate_metrics(kmeans_metrics, knn_metrics)
        file_manager.save_metrics(combined_metrics)
        logger.info(f"All results saved to {config.results_folder}/")

        # Step 9: Print summary
        logger.info("\n[Step 12/12] Pipeline complete!")
        summary = format_metrics_summary(combined_metrics)
        print("\n" + summary)

        logger.info(f"\nResults saved with timestamp: {file_manager.timestamp}")
        logger.info("Pipeline execution successful!")

    except Exception as e:
        logger.error(f"\nPipeline failed with error: {e}", exc_info=True)
        print(f"\nError: {e}")
        print("Check the log file for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
