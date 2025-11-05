# Product Requirements Document (PRD)
## K-Means and KNN Classification System for Sentence Embeddings

---

## Document Information
- **Product Name**: Sentence Classification Visualization System
- **Version**: 1.0
- **Date**: November 4, 2025
- **Product Manager**: Data Science Engineer
- **Project Type**: Machine Learning Classification & Visualization

---

## Executive Summary

This project implements an automated sentence classification system using K-Means clustering and KNN (K-Nearest Neighbors) algorithms. The system generates categorized sentences across three domains (animals, music, food), converts them to normalized vector embeddings using LLM-based agents, and visualizes classification results with color-coded groups.

---

## Product Objectives

### Primary Goals
1. Demonstrate unsupervised learning (K-Means) and supervised learning (KNN) on text data
2. Visualize high-dimensional sentence embeddings in interpretable 2D/3D space
3. Compare clustering vs classification performance on domain-specific text
4. Create reproducible, well-documented ML pipeline for educational purposes

### Success Metrics
- Accurate clustering of sentences into their respective domains (>85% accuracy)
- Clear visual separation of categories in visualization
- Consistent KNN classification performance on new data
- Clean, maintainable codebase with proper documentation

---

## User Requirements

### Input Requirements
- **User Input**: Number of sentences per group (integer, range: 5-100 recommended)
- **Validation**: System must validate input is positive integer
- **Default Value**: 10 sentences per group if no input provided

### Output Requirements
- All results stored in `results/` folder with timestamps
- Two visualization files generated:
  1. K-Means clustering results
  2. KNN classification results
- Each visualization includes:
  - Color-coded scatter plot
  - Legend mapping colors to categories
  - Algorithm name and parameters used
  - Timestamp and metadata

---

## Functional Requirements

### FR-1: Sentence Generation System

#### FR-1.1: Initial Training Data Generation
**Input**: User-specified count (n) per category
**Output**: 3n sentences distributed across 3 categories

| Category | Color Code | Subject Matter | Examples |
|----------|-----------|----------------|----------|
| Animals  | Green (#00FF00) | Wildlife, pets, zoology | "The elephant trumpeted loudly", "Cats purr when content" |
| Music    | Blue (#0000FF) | Instruments, genres, performance | "The piano melody was haunting", "Jazz improvisation requires skill" |
| Food     | Red (#FF0000) | Cuisine, ingredients, cooking | "Fresh pasta tastes amazing", "Chocolate contains antioxidants" |

**Requirements**:
- Each sentence must be 5-20 words long
- Sentences must be grammatically correct
- Clear thematic association with assigned category
- Diverse vocabulary within each category

#### FR-1.2: Test Data Generation
**Input**: Same count (n) per category from user input
**Output**: 3n NEW sentences (not duplicates of training data)

**Requirements**:
- Must follow same category/color assignment
- Must be distinct from training sentences
- Similar complexity and length distribution

---

### FR-2: Vector Embedding System

#### FR-2.1: LLM-Based Vectorization Agent
**Technology**: Use LLM API (OpenAI, Anthropic, or local model) for embedding generation

**Requirements**:
- Dedicated agent/module for sentence-to-vector conversion
- Consistent embedding dimension (e.g., 768, 1536 dimensions)
- Deterministic output for same input (set seed if possible)
- Error handling for API failures/timeouts

#### FR-2.2: Vector Normalization
**CRITICAL REQUIREMENT**: All vectors MUST be normalized before classification

**Implementation**:
- L2 normalization (unit vectors)
- Applied to both training and test vectors
- Verification step to ensure ||v|| = 1 for all vectors

**Formula**: `v_normalized = v / ||v||₂`

**Validation Check**:
```python
assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0), "Vectors not normalized!"
```

---

### FR-3: K-Means Clustering

#### FR-3.1: Algorithm Configuration
**Parameters**:
- Number of clusters (k): 3 (fixed, matching category count)
- Initialization: k-means++ (for better convergence)
- Max iterations: 300
- Random state: Fixed seed for reproducibility
- Convergence tolerance: 1e-4

#### FR-3.2: Processing Pipeline
1. Load all training vectors (3n vectors)
2. Verify normalization
3. Run K-Means algorithm
4. Assign cluster labels to each vector
5. Calculate cluster centers
6. Compute silhouette score and inertia metrics

#### FR-3.3: Output Requirements
- Cluster assignments for each sentence
- Cluster centers (centroids)
- Performance metrics (silhouette score, inertia)
- Confusion matrix comparing clusters to true labels

---

### FR-4: KNN Classification

#### FR-4.1: Algorithm Configuration
**Parameters**:
- Number of neighbors (k): 5 (configurable, odd number preferred)
- Distance metric: Euclidean (on normalized vectors)
- Weighting: Distance-weighted voting
- Algorithm: Auto (ball_tree, kd_tree, or brute)

#### FR-4.2: Processing Pipeline
1. Train KNN model on K-Means results (or original labels)
2. Load test vectors (3n new vectors)
3. Verify normalization of test vectors
4. Predict categories for test data
5. Calculate classification metrics

#### FR-4.3: Output Requirements
- Predicted labels for all test sentences
- Confidence scores per prediction
- Classification report (precision, recall, F1-score)
- Confusion matrix

---

### FR-5: Visualization System

#### FR-5.1: Dimensionality Reduction
**Requirement**: Reduce high-dimensional vectors to 2D for visualization

**Method**: Use one of the following:
- t-SNE (t-distributed Stochastic Neighbor Embedding)
- UMAP (Uniform Manifold Approximation and Projection) - preferred for speed
- PCA (Principal Component Analysis) - baseline

**Parameters**:
- Preserve relative distances
- Random state for reproducibility
- Perplexity/n_neighbors tuned for dataset size

#### FR-5.2: K-Means Visualization
**Filename Format**: `results/kmeans_clustering_YYYYMMDD_HHMMSS.png`

**Required Elements**:
1. **Scatter Plot**:
   - X-axis: First dimension after reduction
   - Y-axis: Second dimension after reduction
   - Point size: 100 (adjustable for readability)
   - Alpha: 0.7 (semi-transparent for overlap visibility)

2. **Color Mapping**:
   - Animals: Green (#00FF00)
   - Music: Blue (#0000FF)
   - Food: Red (#FF0000)
   - Use original category colors (not cluster assignments)

3. **Legend**:
   - Category name + color
   - Sample count per category
   - Format: "Animals (Green) - n samples"

4. **Title**: "K-Means Clustering Results (k=3)"

5. **Metadata Box** (bottom or side):
   ```
   Algorithm: K-Means
   Clusters: 3
   Samples: [total count]
   Normalization: L2
   Dimensionality Reduction: [method]
   Date: [timestamp]
   Silhouette Score: [value]
   ```

6. **Cluster Centers**: Mark with large 'X' symbols

#### FR-5.3: KNN Classification Visualization
**Filename Format**: `results/knn_classification_YYYYMMDD_HHMMSS.png`

**Required Elements**:
1. **Scatter Plot** (same specs as FR-5.2)

2. **Dual Color System** (optional enhancement):
   - Marker fill: Predicted category color
   - Marker edge: True category color
   - Edge width: 2 (to show misclassifications)

3. **Legend**:
   - Category name + color + counts
   - Include prediction accuracy per category
   - Format: "Animals (Green) - n samples, accuracy: xx%"

4. **Title**: "KNN Classification Results (k=5)"

5. **Metadata Box**:
   ```
   Algorithm: K-Nearest Neighbors
   K value: 5
   Training samples: [count]
   Test samples: [count]
   Normalization: L2
   Overall Accuracy: [value]
   Date: [timestamp]
   ```

#### FR-5.4: Additional Visualization Requirements
- High resolution: 300 DPI minimum
- Size: 12x8 inches (landscape)
- Grid: Light gray, alpha 0.3
- Font: Arial or Helvetica, 10-12pt
- Save formats: PNG (required), SVG (optional for scalability)

---

### FR-6: Results Storage

#### FR-6.1: Folder Structure
```
results/
├── kmeans_clustering_YYYYMMDD_HHMMSS.png
├── knn_classification_YYYYMMDD_HHMMSS.png
├── training_data_YYYYMMDD_HHMMSS.json
├── test_data_YYYYMMDD_HHMMSS.json
├── vectors_YYYYMMDD_HHMMSS.npz
├── metrics_YYYYMMDD_HHMMSS.json
└── run_log_YYYYMMDD_HHMMSS.txt
```

#### FR-6.2: Data Files

**training_data_YYYYMMDD_HHMMSS.json**:
```json
{
  "animals": ["sentence1", "sentence2", ...],
  "music": ["sentence1", "sentence2", ...],
  "food": ["sentence1", "sentence2", ...],
  "metadata": {
    "count_per_category": 10,
    "generation_timestamp": "2025-11-04T10:30:00Z",
    "total_sentences": 30
  }
}
```

**test_data_YYYYMMDD_HHMMSS.json**: Same structure as training data

**vectors_YYYYMMDD_HHMMSS.npz**:
```python
{
  "training_vectors": np.array,  # Shape: (3n, embedding_dim)
  "test_vectors": np.array,      # Shape: (3n, embedding_dim)
  "training_labels": np.array,   # Shape: (3n,)
  "test_labels": np.array,       # Shape: (3n,)
  "normalization_applied": True
}
```

**metrics_YYYYMMDD_HHMMSS.json**:
```json
{
  "kmeans": {
    "silhouette_score": 0.xx,
    "inertia": xxx.xx,
    "cluster_sizes": [x, x, x],
    "confusion_matrix": [[...], [...], [...]]
  },
  "knn": {
    "accuracy": 0.xx,
    "precision": {"animals": 0.xx, "music": 0.xx, "food": 0.xx},
    "recall": {"animals": 0.xx, "music": 0.xx, "food": 0.xx},
    "f1_score": {"animals": 0.xx, "music": 0.xx, "food": 0.xx},
    "confusion_matrix": [[...], [...], [...]]
  }
}
```

**run_log_YYYYMMDD_HHMMSS.txt**: Detailed execution log with timestamps

---

## Non-Functional Requirements

### NFR-1: Performance
- Maximum execution time: 5 minutes for 30 sentences per category
- LLM API response time: < 3 seconds per batch of 10 sentences
- Memory usage: < 2GB RAM for typical workloads

### NFR-2: Reliability
- Graceful handling of API failures with retry logic (3 attempts)
- Data validation at each pipeline stage
- Automatic backup of intermediate results
- Error logging with stack traces

### NFR-3: Usability
- Clear command-line interface with progress indicators
- Informative error messages with suggested fixes
- README with setup instructions and examples
- Help command showing all options

### NFR-4: Maintainability
- Modular code structure (separate files for generation, vectorization, clustering, visualization)
- Type hints for all functions
- Docstrings following Google/NumPy style
- Unit tests for core functions (>70% coverage)

### NFR-5: Scalability
- Support for 5-100 sentences per category
- Configurable batch size for API calls
- Optional caching of generated vectors

---

## Technical Stack

### Required Libraries
```
Python >= 3.8

Core ML:
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0

LLM Integration:
- openai >= 1.0.0 OR anthropic >= 0.8.0
- tiktoken (for token counting)

Visualization:
- matplotlib >= 3.5.0
- seaborn >= 0.11.0 (optional, for enhanced plots)
- umap-learn >= 0.5.0 OR scikit-learn (for t-SNE)

Utilities:
- python-dotenv (for API key management)
- tqdm (for progress bars)
- pytest (for testing)
```

### Environment Setup
- API keys stored in `.env` file (not version controlled)
- Virtual environment recommended
- Requirements.txt for reproducibility

---

## Data Flow Diagram

```
User Input (n)
    ↓
[Sentence Generator]
    ↓
Training Sentences (3n) ----→ [LLM Vectorization Agent] ----→ Training Vectors (3n)
    ↓                                                               ↓
Test Sentences (3n) --------→ [LLM Vectorization Agent] ----→ Test Vectors (3n)
                                                                    ↓
                                                            [Normalization Check]
                                                                    ↓
Training Vectors (normalized) ----→ [K-Means Clustering] ----→ Cluster Labels
    ↓                                       ↓                        ↓
    |                               [Visualization 1] ----→ PNG + Metadata
    |                                       ↓
    |                                  results/folder
    ↓
[KNN Training] ←--- Cluster Labels
    ↓
[KNN Prediction] ←--- Test Vectors (normalized)
    ↓
Test Predictions ----→ [Visualization 2] ----→ PNG + Metadata
    ↓                         ↓
[Metrics Calculation]    results/folder
    ↓
results/metrics.json
```

---

## Validation & Quality Assurance

### Pre-Classification Validation Checklist
- [ ] All vectors have consistent dimensions
- [ ] All vectors are L2 normalized (||v|| = 1)
- [ ] No NaN or Inf values in vectors
- [ ] Sentence count matches expected (3n for each dataset)
- [ ] Categories properly labeled (0: animals, 1: music, 2: food)

### Post-Processing Validation
- [ ] Visualization files created successfully
- [ ] Metadata matches actual data
- [ ] Color mapping is correct
- [ ] All result files present in results/ folder
- [ ] Metrics are within reasonable ranges (0-1 for scores)

---

## Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| LLM API rate limiting | High | Medium | Implement exponential backoff, batch processing |
| Poor clustering separation | Medium | Low | Use quality sentence generation, verify embeddings |
| Visualization clarity issues | Low | Medium | Test with various sample sizes, adjust parameters |
| High dimensional curse | Medium | Low | Use robust dimensionality reduction (UMAP/t-SNE) |
| Inconsistent embeddings | High | Low | Use deterministic seed, validate consistency |

---

## Future Enhancements (Out of Scope for v1.0)

1. **Multi-Language Support**: Extend to non-English sentences
2. **Custom Categories**: Allow users to define their own categories
3. **Interactive Visualization**: Web-based dashboard with zoom/pan
4. **Comparative Analysis**: Compare different embedding models
5. **Hyperparameter Tuning**: Automatic optimization of k, distance metrics
6. **Real-Time Mode**: Live classification of user-input sentences
7. **Export Options**: PDF reports with detailed analysis
8. **A/B Testing**: Compare K-Means vs other clustering algorithms

---

## Acceptance Criteria

### Must Have (v1.0)
- ✓ User can input sentence count via CLI
- ✓ System generates 3n training + 3n test sentences
- ✓ All vectors are normalized before classification
- ✓ K-Means clustering produces visualization with metadata
- ✓ KNN classification produces visualization with metadata
- ✓ All results saved to results/ folder with timestamps
- ✓ Color mapping is correct (green/blue/red for animals/music/food)
- ✓ Legends include category names, colors, and sample counts

### Should Have
- ✓ Metrics exported to JSON file
- ✓ Progress indicators during execution
- ✓ Error handling for common failures
- ✓ README with usage examples

### Nice to Have
- Unit tests for core functions
- Command-line arguments for advanced configuration
- SVG export option for visualizations
- Comparison metrics between K-Means and KNN

---

## Glossary

- **Embedding**: Dense vector representation of text in continuous space
- **L2 Normalization**: Scaling vector to unit length (||v|| = 1)
- **Silhouette Score**: Metric measuring cluster quality (-1 to 1, higher is better)
- **Inertia**: Sum of squared distances to nearest cluster center
- **t-SNE**: Non-linear dimensionality reduction preserving local structure
- **UMAP**: Faster alternative to t-SNE, preserves global structure better
- **K-Means++**: Improved initialization method for K-Means
- **Distance-weighted voting**: KNN votes weighted by inverse distance

---

## Appendix A: Sample Sentences

### Animals (Green)
1. "The leopard stalked silently through the tall grass at dawn."
2. "Dolphins communicate using complex patterns of clicks and whistles."
3. "Hummingbirds flap their wings up to eighty times per second."

### Music (Blue)
1. "The saxophone solo brought the audience to their feet in applause."
2. "Classical symphonies often feature four distinct movements with varying tempos."
3. "Electronic music producers layer synthesizers to create rich atmospheric sounds."

### Food (Red)
1. "Artisanal bread requires proper fermentation for optimal flavor development."
2. "Sushi chefs train for years to master the art of rice preparation."
3. "Freshly ground black pepper enhances the flavor of most savory dishes."

---

## Appendix B: Configuration Template

**config.yaml**
```yaml
# User Input
sentences_per_category: 10

# LLM Configuration
llm:
  provider: "openai"  # or "anthropic"
  model: "text-embedding-3-small"
  api_key_env: "OPENAI_API_KEY"
  batch_size: 10
  timeout: 30

# K-Means Configuration
kmeans:
  n_clusters: 3
  init: "k-means++"
  max_iter: 300
  n_init: 10
  random_state: 42

# KNN Configuration
knn:
  n_neighbors: 5
  weights: "distance"
  metric: "euclidean"
  algorithm: "auto"

# Visualization Configuration
visualization:
  reduction_method: "umap"  # or "tsne", "pca"
  figure_size: [12, 8]
  dpi: 300
  point_size: 100
  alpha: 0.7

# Colors
colors:
  animals: "#00FF00"
  music: "#0000FF"
  food: "#FF0000"

# Output Configuration
output:
  results_folder: "results"
  save_intermediate: true
  export_svg: false
```

---

## Sign-Off

**Product Manager**: ______________________ Date: __________

**Engineering Lead**: ______________________ Date: __________

**QA Lead**: ______________________________ Date: __________

---

*End of Document*