# System Design Document
## K-Means & KNN Sentence Classification System

---

## Document Overview

This document describes the architecture, design decisions, and operational flow of the sentence classification system. The system uses OpenAI embeddings, K-Means clustering, and KNN classification to categorize sentences into three domains: animals, music, and food.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Project Structure](#project-structure)
3. [Component Design](#component-design)
4. [Data Flow](#data-flow)
5. [Operational Flowcharts](#operational-flowcharts)
6. [Design Decisions](#design-decisions)
7. [File Size Constraints](#file-size-constraints)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
│                  (Command Line Interface)                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                           │
│                         (main.py)                                │
│  • Parse CLI arguments                                           │
│  • Load configuration                                            │
│  • Coordinate pipeline execution                                 │
│  • Handle errors and logging                                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ↓               ↓               ↓
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  DATA LAYER     │ │  AGENT LAYER    │ │  ML LAYER       │
│                 │ │                 │ │                 │
│ • Generator     │ │ • Vectorization │ │ • K-Means       │
│ • Storage       │ │   Agent         │ │ • KNN           │
│ • Validation    │ │ • OpenAI API    │ │ • Metrics       │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ↓
                  ┌──────────────────────┐
                  │  VISUALIZATION LAYER │
                  │                      │
                  │ • UMAP Reduction     │
                  │ • Matplotlib Plots   │
                  │ • Metadata Display   │
                  └──────────┬───────────┘
                             │
                             ↓
                  ┌──────────────────────┐
                  │   RESULTS STORAGE    │
                  │                      │
                  │ • PNG visualizations │
                  │ • JSON data files    │
                  │ • NPZ vector arrays  │
                  │ • Execution logs     │
                  └──────────────────────┘
```

### Layer Responsibilities

#### 1. User Interface Layer
- **Purpose**: Accept user input and display results
- **Components**: CLI argument parser
- **Technology**: argparse

#### 2. Orchestration Layer
- **Purpose**: Coordinate all components and manage pipeline flow
- **Components**: main.py, config.py
- **Responsibilities**:
  - Load configuration from .env
  - Initialize all modules
  - Execute pipeline steps in order
  - Handle errors gracefully
  - Log all operations

#### 3. Data Layer
- **Purpose**: Generate and manage sentence data
- **Components**: sentence_generator.py, data_validator.py
- **Responsibilities**:
  - Generate categorized sentences
  - Validate data quality
  - Store/load training and test sets

#### 4. Agent Layer
- **Purpose**: Convert sentences to vector embeddings
- **Components**: agents/vectorization_agent.py
- **Responsibilities**:
  - Interface with OpenAI API
  - Handle API errors and retries
  - Batch processing for efficiency
  - Vector normalization

#### 5. ML Layer
- **Purpose**: Apply machine learning algorithms
- **Components**: clustering.py, classification.py, metrics.py
- **Responsibilities**:
  - K-Means clustering
  - KNN classification
  - Metric calculation
  - Performance evaluation

#### 6. Visualization Layer
- **Purpose**: Create visual representations of results
- **Components**: visualization.py, plotting.py
- **Responsibilities**:
  - Dimensionality reduction (UMAP)
  - Scatter plot generation
  - Metadata overlay
  - Color mapping

#### 7. Storage Layer
- **Purpose**: Persist all results
- **Components**: storage.py, file_utils.py
- **Responsibilities**:
  - Save visualizations
  - Export data files
  - Store metrics
  - Generate logs

---

## Project Structure

### Directory Layout

```
Ex16-KNN-AlgoPractice/
│
├── .env                          # Environment variables (API key)
├── .env.example                  # Template for .env
├── .gitignore                    # Git ignore rules
├── pyproject.toml                # uv project configuration
├── README.md                     # Project documentation
├── PRD.md                        # Product requirements
├── planning.md                   # Implementation plan
├── tasks.json                    # Task breakdown
├── Claude.md                     # Claude's working context
├── DESIGN.md                     # This file
│
├── agents/                       # AGENT IMPLEMENTATIONS (Separate folder)
│   ├── __init__.py
│   ├── vectorization_agent.py   # OpenAI embedding agent (~150 lines)
│   └── agent_utils.py           # Agent helper functions (~100 lines)
│
├── src/                          # MAIN PROGRAM IMPLEMENTATIONS
│   ├── __init__.py
│   │
│   ├── core/                     # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py            # Configuration management (~120 lines)
│   │   └── utils.py             # General utilities (~150 lines)
│   │
│   ├── data/                     # Data generation and management
│   │   ├── __init__.py
│   │   ├── sentence_generator.py  # Sentence generation (~180 lines)
│   │   └── data_validator.py      # Data validation (~100 lines)
│   │
│   ├── ml/                       # Machine learning components
│   │   ├── __init__.py
│   │   ├── clustering.py        # K-Means implementation (~150 lines)
│   │   ├── classification.py    # KNN implementation (~150 lines)
│   │   └── metrics.py           # Metric calculation (~120 lines)
│   │
│   ├── visualization/            # Visualization components
│   │   ├── __init__.py
│   │   ├── dimensionality_reduction.py  # UMAP/t-SNE (~100 lines)
│   │   ├── plotting.py          # Matplotlib plots (~180 lines)
│   │   └── color_utils.py       # Color management (~80 lines)
│   │
│   ├── storage/                  # Results storage
│   │   ├── __init__.py
│   │   ├── file_manager.py      # File operations (~150 lines)
│   │   └── export_utils.py      # Export helpers (~100 lines)
│   │
│   └── main.py                   # Main orchestration (~180 lines)
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── conftest.py              # Pytest fixtures (~100 lines)
│   ├── test_agents/             # Agent tests
│   │   └── test_vectorization_agent.py (~150 lines)
│   ├── test_data/               # Data tests
│   │   └── test_sentence_generator.py (~150 lines)
│   ├── test_ml/                 # ML tests
│   │   ├── test_clustering.py (~120 lines)
│   │   └── test_classification.py (~120 lines)
│   ├── test_visualization/      # Visualization tests
│   │   └── test_plotting.py (~100 lines)
│   └── test_integration.py      # End-to-end test (~150 lines)
│
└── results/                      # Generated outputs (git-ignored)
    ├── kmeans_clustering_*.png
    ├── knn_classification_*.png
    ├── training_data_*.json
    ├── test_data_*.json
    ├── vectors_*.npz
    ├── metrics_*.json
    └── run_log_*.txt
```

### File Size Constraints

**Rule**: No file exceeds 200 lines

| File | Max Lines | Purpose |
|------|-----------|---------|
| `agents/vectorization_agent.py` | ~150 | OpenAI API integration |
| `src/core/config.py` | ~120 | Configuration management |
| `src/core/utils.py` | ~150 | General utilities |
| `src/data/sentence_generator.py` | ~180 | Sentence generation |
| `src/data/data_validator.py` | ~100 | Validation logic |
| `src/ml/clustering.py` | ~150 | K-Means clustering |
| `src/ml/classification.py` | ~150 | KNN classification |
| `src/ml/metrics.py` | ~120 | Metric calculation |
| `src/visualization/dimensionality_reduction.py` | ~100 | UMAP/t-SNE |
| `src/visualization/plotting.py` | ~180 | Matplotlib plotting |
| `src/visualization/color_utils.py` | ~80 | Color management |
| `src/storage/file_manager.py` | ~150 | File I/O |
| `src/storage/export_utils.py` | ~100 | Export operations |
| `src/main.py` | ~180 | Pipeline orchestration |

**Total Lines**: ~2,000 lines across 14 modules

---

## Component Design

### 1. Vectorization Agent (agents/vectorization_agent.py)

**Purpose**: Convert text to vector embeddings using OpenAI API

**Class**: `VectorizationAgent`

**Key Methods**:
```python
class VectorizationAgent:
    def __init__(config: Config)
        # Initialize OpenAI client with API key

    def vectorize(sentences: List[str]) -> np.ndarray
        # Convert sentences to vectors
        # Returns: normalized vectors (n_samples, 1536)

    def vectorize_batch(sentences: List[str], batch_size: int) -> np.ndarray
        # Process large lists in batches
        # Shows progress bar with tqdm

    def _call_api_with_retry(sentences: List[str], max_attempts: int) -> np.ndarray
        # Call OpenAI with exponential backoff
        # Handles rate limits, connection errors
```

**Error Handling**:
- Rate limit errors: Exponential backoff (1s, 2s, 4s)
- Authentication errors: Clear message to check API key
- Connection errors: Retry with backoff
- API errors: Log and raise with context

**Comments Standard**:
```python
def vectorize(self, sentences: List[str], normalize: bool = True) -> np.ndarray:
    """
    Convert list of sentences to normalized vector embeddings.

    Args:
        sentences: List of text strings to embed
        normalize: Whether to apply L2 normalization (default: True)

    Returns:
        NumPy array of shape (n_sentences, 1536) with normalized vectors

    Raises:
        ValueError: If sentences list is empty
        APIError: If OpenAI API call fails after retries

    Example:
        >>> agent = VectorizationAgent(config)
        >>> vectors = agent.vectorize(["The cat sits", "Dogs bark"])
        >>> vectors.shape
        (2, 1536)
    """
    # Validate input
    if not sentences:
        raise ValueError("Sentences list cannot be empty")

    # Call OpenAI API with retry logic
    embeddings = self._call_api_with_retry(sentences)

    # Apply L2 normalization if requested
    if normalize:
        embeddings = normalize_vectors(embeddings)

    return embeddings
```

### 2. Sentence Generator (src/data/sentence_generator.py)

**Purpose**: Generate categorized sentences for training and testing

**Class**: `SentenceGenerator`

**Categories**:
- **Animals** (Green): Wildlife, pets, behavior
- **Music** (Blue): Instruments, genres, performance
- **Food** (Red): Cuisine, ingredients, cooking

**Key Methods**:
```python
class SentenceGenerator:
    def __init__(config: Config)
        # Load templates and vocabulary

    def generate_training_set(n_per_category: int) -> Dict[str, List[str]]
        # Generate training sentences
        # Returns: {"animals": [...], "music": [...], "food": [...]}

    def generate_test_set(n_per_category: int) -> Dict[str, List[str]]
        # Generate test sentences (different from training)

    def _generate_category_sentences(category: str, count: int) -> List[str]
        # Generate sentences for specific category
        # Uses template-based generation with randomization
```

**Template Examples**:
```python
ANIMAL_TEMPLATES = [
    "The {animal} {verb} {adverb} in the {location}",
    "{animal_plural} are known for their {adjective} {trait}",
    "A {color} {animal} {verb} near the {location}"
]

MUSIC_TEMPLATES = [
    "The {instrument} produced a {adjective} {sound}",
    "{musician_type} often {verb} with great {emotion}",
    "{genre} music features {adjective} rhythms"
]
```

### 3. K-Means Clustering (src/ml/clustering.py)

**Purpose**: Unsupervised clustering of sentence vectors

**Class**: `KMeansClustering`

**Key Methods**:
```python
class KMeansClustering:
    def __init__(config: Config)
        # Initialize K-Means with k=3

    def fit(vectors: np.ndarray) -> np.ndarray
        # Fit clustering and return labels

    def get_metrics() -> Dict[str, float]
        # Return silhouette score, inertia

    def get_cluster_centers() -> np.ndarray
        # Return cluster centroids
```

**Configuration**:
```python
KMeans(
    n_clusters=3,           # Fixed: 3 categories
    init='k-means++',       # Smart initialization
    max_iter=300,           # Maximum iterations
    n_init=10,              # Number of initializations
    random_state=42         # Reproducibility
)
```

### 4. KNN Classification (src/ml/classification.py)

**Purpose**: Supervised classification of new sentences

**Class**: `KNNClassifier`

**Key Methods**:
```python
class KNNClassifier:
    def __init__(config: Config)
        # Initialize KNN with k=5

    def train(X_train: np.ndarray, y_train: np.ndarray) -> None
        # Train classifier on labeled data

    def predict(X_test: np.ndarray) -> np.ndarray
        # Predict labels for test data

    def predict_proba(X_test: np.ndarray) -> np.ndarray
        # Return class probabilities

    def get_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict
        # Calculate accuracy, precision, recall, F1
```

### 5. Visualization (src/visualization/plotting.py)

**Purpose**: Create publication-quality plots with metadata

**Key Functions**:
```python
def create_kmeans_plot(
    vectors_2d: np.ndarray,
    labels: np.ndarray,
    colors: Dict[str, str],
    metrics: Dict,
    output_path: str
) -> None:
    # Create K-Means scatter plot with:
    # - Color-coded points by category
    # - Cluster centers marked
    # - Legend with counts
    # - Metadata box with metrics

def create_knn_plot(
    vectors_2d: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    colors: Dict[str, str],
    metrics: Dict,
    output_path: str
) -> None:
    # Create KNN scatter plot with:
    # - Color-coded predictions
    # - Misclassifications highlighted
    # - Accuracy per category
    # - Metadata box
```

---

## Data Flow

### Complete Pipeline Flow

```
START
  │
  ├─→ [1. Load Configuration]
  │    - Read .env file for API key
  │    - Set default parameters
  │    - Validate configuration
  │    └─→ Config object
  │
  ├─→ [2. Initialize Components]
  │    - Create SentenceGenerator
  │    - Create VectorizationAgent (OpenAI client)
  │    - Create KMeansClustering
  │    - Create KNNClassifier
  │    - Create Visualizer
  │    └─→ All modules ready
  │
  ├─→ [3. Generate Training Data]
  │    - Generate N sentences per category
  │    - Categories: animals, music, food
  │    - Validate: no duplicates, proper length
  │    └─→ training_sentences: Dict[str, List[str]] (3N sentences)
  │
  ├─→ [4. Generate Test Data]
  │    - Generate N NEW sentences per category
  │    - Ensure disjoint from training set
  │    - Validate quality
  │    └─→ test_sentences: Dict[str, List[str]] (3N sentences)
  │
  ├─→ [5. Vectorize Training Data]
  │    - Call OpenAI API in batches
  │    - Convert sentences to embeddings (1536-dim)
  │    - Apply L2 normalization
  │    - Validate: ||v|| = 1 for all vectors
  │    └─→ training_vectors: np.ndarray (3N, 1536)
  │
  ├─→ [6. Vectorize Test Data]
  │    - Call OpenAI API for test sentences
  │    - Apply same normalization
  │    - Validate normalization
  │    └─→ test_vectors: np.ndarray (3N, 1536)
  │
  ├─→ [7. Run K-Means Clustering]
  │    - Fit K-Means on training_vectors (k=3)
  │    - Get cluster assignments
  │    - Calculate metrics (silhouette, inertia)
  │    - Generate confusion matrix vs true labels
  │    └─→ cluster_labels: np.ndarray (3N,)
  │         cluster_centers: np.ndarray (3, 1536)
  │         kmeans_metrics: Dict
  │
  ├─→ [8. Visualize K-Means Results]
  │    - Reduce training_vectors to 2D (UMAP)
  │    - Create scatter plot with original colors
  │    - Add cluster centers as X markers
  │    - Add legend and metadata box
  │    - Save as PNG with timestamp
  │    └─→ results/kmeans_clustering_YYYYMMDD_HHMMSS.png
  │
  ├─→ [9. Train KNN Classifier]
  │    - Train on training_vectors + true labels
  │    - Use k=5 neighbors, distance weighting
  │    └─→ trained_knn: KNNClassifier
  │
  ├─→ [10. Predict Test Data]
  │    - Predict labels for test_vectors
  │    - Calculate confidence scores
  │    └─→ predictions: np.ndarray (3N,)
  │         probabilities: np.ndarray (3N, 3)
  │
  ├─→ [11. Calculate KNN Metrics]
  │    - Accuracy, precision, recall, F1
  │    - Per-class metrics
  │    - Confusion matrix
  │    └─→ knn_metrics: Dict
  │
  ├─→ [12. Visualize KNN Results]
  │    - Reduce test_vectors to 2D (UMAP)
  │    - Create scatter plot with true colors
  │    - Highlight misclassifications
  │    - Add per-category accuracy to legend
  │    - Add metadata box
  │    - Save as PNG with timestamp
  │    └─→ results/knn_classification_YYYYMMDD_HHMMSS.png
  │
  ├─→ [13. Save All Results]
  │    - Save training_data.json
  │    - Save test_data.json
  │    - Save vectors.npz (training + test)
  │    - Save metrics.json (K-Means + KNN)
  │    - Save execution log
  │    └─→ All files in results/ folder
  │
  └─→ [14. Display Summary]
       - Print K-Means metrics
       - Print KNN metrics
       - Print file paths
       └─→ END (Success)
```

---

## Operational Flowcharts

### Flowchart 1: Main Pipeline Execution

```
┌─────────────────┐
│  Parse CLI Args │
│  --sentences N  │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Load Config     │
│ (.env + defaults)│
└────────┬────────┘
         │
         ↓
┌─────────────────┐     NO      ┌──────────────┐
│ API Key Valid?  │─────────────→│ Error: Check │
└────────┬────────┘              │   .env file  │
         │YES                     └──────────────┘
         ↓
┌─────────────────┐
│ Generate        │
│ Training        │
│ Sentences       │
│ (3N total)      │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Generate        │
│ Test Sentences  │
│ (3N total)      │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Vectorize       │
│ Training Data   │
│ (OpenAI API)    │
└────────┬────────┘
         │
         ↓
┌─────────────────┐     NO      ┌──────────────┐
│ Vectors         │─────────────→│ Normalize    │
│ Normalized?     │              │ Vectors      │
└────────┬────────┘              └──────┬───────┘
         │YES                            │
         └───────────────────────────────┘
         │
         ↓
┌─────────────────┐
│ Vectorize       │
│ Test Data       │
│ (OpenAI API)    │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Run K-Means     │
│ Clustering      │
│ (k=3)           │
└────────┬────────┘
         │
         ├────────────────┐
         │                │
         ↓                ↓
┌─────────────────┐  ┌─────────────────┐
│ Calculate       │  │ Visualize       │
│ K-Means         │  │ K-Means         │
│ Metrics         │  │ Results         │
└────────┬────────┘  └────────┬────────┘
         │                    │
         └────────┬───────────┘
                  │
                  ↓
         ┌─────────────────┐
         │ Train KNN       │
         │ Classifier      │
         │ (k=5)           │
         └────────┬────────┘
                  │
                  ↓
         ┌─────────────────┐
         │ Predict Test    │
         │ Data with KNN   │
         └────────┬────────┘
                  │
                  ├────────────────┐
                  │                │
                  ↓                ↓
         ┌─────────────────┐  ┌─────────────────┐
         │ Calculate       │  │ Visualize       │
         │ KNN Metrics     │  │ KNN Results     │
         └────────┬────────┘  └────────┬────────┘
                  │                    │
                  └────────┬───────────┘
                           │
                           ↓
                  ┌─────────────────┐
                  │ Save All        │
                  │ Results to      │
                  │ results/        │
                  └────────┬────────┘
                           │
                           ↓
                  ┌─────────────────┐
                  │ Print Summary   │
                  │ & File Paths    │
                  └────────┬────────┘
                           │
                           ↓
                        SUCCESS
```

### Flowchart 2: Vectorization Agent Operation

```
┌─────────────────────┐
│ Receive Sentences   │
│ List[str]           │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│ Split into Batches  │
│ (batch_size=100)    │
└──────────┬──────────┘
           │
           ↓
    ┌──────────────┐
    │ For Each     │
    │ Batch        │
    └──────┬───────┘
           │
           ↓
┌─────────────────────┐
│ Call OpenAI API     │
│ embeddings.create() │
└──────────┬──────────┘
           │
           ├─ Rate Limit Error?
           │  └─→ YES → Wait (exponential backoff) → Retry
           │           ↓
           │           NO (max 3 attempts)
           │           ↓
           │     ┌──────────────┐
           │     │ Raise Error  │
           │     └──────────────┘
           │
           ├─ Connection Error?
           │  └─→ YES → Wait & Retry → (same as above)
           │
           ├─ Auth Error?
           │  └─→ YES → ┌──────────────────┐
           │            │ Error: Invalid   │
           │            │ API Key          │
           │            └──────────────────┘
           │
           ↓ SUCCESS
┌─────────────────────┐
│ Extract Embeddings  │
│ from Response       │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│ Convert to NumPy    │
│ Array (float32)     │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│ Collect All Batches │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│ Apply L2            │
│ Normalization       │
│ v_norm = v / ||v||  │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐     NO    ┌──────────────┐
│ Validate:           │───────────→│ Raise        │
│ ||v|| ≈ 1?          │            │ ValueError   │
└──────────┬──────────┘            └──────────────┘
           │YES
           ↓
┌─────────────────────┐
│ Return Normalized   │
│ Vectors             │
│ np.ndarray          │
└─────────────────────┘
```

### Flowchart 3: Visualization Creation

```
┌─────────────────────┐
│ Input: Vectors      │
│ (n_samples, 1536)   │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│ Dimensionality      │
│ Reduction (UMAP)    │
│ 1536D → 2D          │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│ Create Figure       │
│ (12x8 inches, 300dpi)│
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│ For Each Category:  │
│ • Animals (Green)   │
│ • Music (Blue)      │
│ • Food (Red)        │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│ Plot Scatter Points │
│ • Size: 100         │
│ • Alpha: 0.7        │
│ • EdgeColor: black  │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│ Add Legend          │
│ • Category name     │
│ • Color indicator   │
│ • Sample count      │
│ • (KNN: accuracy)   │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│ Add Title           │
│ (Algorithm name)    │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│ Create Metadata Box │
│ • Algorithm info    │
│ • Parameters        │
│ • Metrics           │
│ • Timestamp         │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│ Position Metadata   │
│ (top-left corner)   │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│ Add Grid & Styling  │
│ • Grid: alpha=0.3   │
│ • Axis labels       │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│ Generate Filename   │
│ {alg}_{timestamp}.png│
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│ Save to results/    │
│ (PNG, 300 DPI)      │
└──────────┬──────────┘
           │
           ↓
        SUCCESS
```

---

## Design Decisions

### 1. Why Separate agents/ and src/ Folders?

**Rationale**:
- **Agents** represent external system integrations (OpenAI API)
- **src/** contains core business logic
- Clear separation of concerns
- Easier to mock agents in tests
- Agents can be reused across projects

### 2. Why Template-Based Sentence Generation?

**Alternatives Considered**:
1. Use GPT-4 to generate sentences (costly, requires API calls)
2. Use pre-defined sentence list (no variety)
3. Template-based with randomization (chosen)

**Decision**: Template-based generation
- **Pros**: Free, fast, deterministic, easily ensures train/test separation
- **Cons**: Less natural variety than LLM generation
- **Trade-off**: Acceptable for educational/demo purposes

### 3. Why UMAP for Dimensionality Reduction?

**Alternatives Considered**:
1. PCA (fast but linear)
2. t-SNE (good quality but slow)
3. UMAP (chosen)

**Decision**: UMAP as primary, with fallback options
- **Pros**: Preserves both local and global structure, faster than t-SNE
- **Cons**: Requires additional dependency
- **Fallback**: t-SNE for comparison, PCA for speed

### 4. Why L2 Normalization?

**Rationale**:
- KNN with Euclidean distance is sensitive to vector magnitude
- Normalization ensures fair distance calculations
- OpenAI embeddings may have varying magnitudes
- Standard practice in similarity search

**Validation**: Assert all vectors have unit norm (||v|| = 1)

### 5. Why JSON for Data Storage?

**Alternatives**: CSV, Pickle, HDF5

**Decision**: JSON for sentences, NPZ for vectors
- **JSON Pros**: Human-readable, language-agnostic, version-control friendly
- **NPZ Pros**: Efficient for NumPy arrays, preserves data types
- **Trade-off**: Slight overhead vs pickle, but better interoperability

### 6. Why 200-Line File Limit?

**Rationale**:
- Improves code readability
- Encourages modular design
- Easier to test individual components
- Reduces cognitive load
- Follows single responsibility principle

**Strategy**: Split by functionality into submodules

---

## Comments and Documentation Standards

### Function Comment Template

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    Brief one-line description of function purpose.

    More detailed explanation if needed, describing what the function does,
    why it exists, and any important implementation details.

    Args:
        param1: Description of first parameter
        param2: Description of second parameter

    Returns:
        Description of return value and its structure

    Raises:
        ExceptionType: When and why this exception is raised

    Example:
        >>> result = function_name(value1, value2)
        >>> print(result)
        expected_output

    Note:
        Any important notes about usage, performance, or limitations
    """
    # Implementation with inline comments
    pass
```

### Inline Comment Guidelines

```python
# Step 1: Validate input parameters
if not sentences:
    raise ValueError("Sentences list cannot be empty")

# Step 2: Call API with retry logic for robustness
# We use exponential backoff to handle rate limits gracefully
embeddings = self._call_api_with_retry(sentences, max_attempts=3)

# Step 3: Normalize vectors to unit length for fair distance comparison
# Formula: v_normalized = v / ||v||₂
normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Step 4: Validate normalization succeeded
# All vectors should have L2 norm approximately equal to 1.0
validate_normalization(normalized_embeddings, tolerance=1e-5)

return normalized_embeddings
```

### Class Comment Template

```python
class ClassName:
    """
    Brief one-line description of class purpose.

    Detailed description of what the class represents, its responsibilities,
    and how it fits into the overall system architecture.

    Attributes:
        attribute1: Description of instance variable
        attribute2: Description of another attribute

    Example:
        >>> obj = ClassName(config)
        >>> result = obj.method_name(arg)

    Note:
        Any important notes about thread-safety, state management, etc.
    """
    pass
```

---

## Error Handling Strategy

### Exception Hierarchy

```
Exception
│
├── APIError (Base for API-related errors)
│   ├── RateLimitError (429 from OpenAI)
│   ├── AuthenticationError (401 from OpenAI)
│   └── ConnectionError (Network issues)
│
├── ValidationError (Base for validation errors)
│   ├── NormalizationError (Vectors not normalized)
│   ├── DataQualityError (Sentences invalid)
│   └── ConfigurationError (Invalid config)
│
└── StorageError (File I/O issues)
    ├── FileNotFoundError
    └── PermissionError
```

### Retry Strategy

```python
# Exponential backoff for transient errors
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds
BACKOFF_FACTOR = 2

for attempt in range(MAX_RETRIES):
    try:
        result = api_call()
        return result
    except RateLimitError:
        if attempt < MAX_RETRIES - 1:
            delay = BASE_DELAY * (BACKOFF_FACTOR ** attempt)
            logger.warning(f"Rate limited. Retrying in {delay}s...")
            time.sleep(delay)
        else:
            logger.error("Max retries exceeded")
            raise
```

---

## Testing Strategy

### Test Structure

```
tests/
├── Unit Tests (test individual functions)
│   ├── Mock external dependencies (OpenAI API)
│   ├── Test edge cases
│   └── Assert expected behavior
│
├── Integration Tests (test component interactions)
│   ├── Test data flow between modules
│   ├── Use small datasets (5 sentences per category)
│   └── Verify end-to-end pipeline
│
└── Validation Tests (test constraints)
    ├── Verify normalization
    ├── Check file outputs
    └── Validate metrics ranges
```

### Coverage Goals

- **Unit Test Coverage**: >70%
- **Critical Paths**: 100% (vectorization, normalization, clustering, KNN)
- **Integration**: Full pipeline tested

---

## Performance Targets

| Operation | Target Time | Notes |
|-----------|-------------|-------|
| Sentence Generation (30 sentences) | <1s | Template-based, fast |
| Vectorization (30 sentences) | 3-5s | Depends on API latency |
| K-Means Clustering | <1s | Small dataset (30 vectors) |
| KNN Training | <0.5s | Fast for small datasets |
| KNN Prediction | <0.5s | Distance calculations |
| UMAP Reduction | 2-3s | Depends on n_neighbors |
| Visualization | 1-2s | Matplotlib rendering |
| **Total Pipeline** | **10-15s** | For 30 total sentences |

---

## Security Considerations

### API Key Protection

1. **Never commit .env** to version control
2. **.gitignore** includes `.env` pattern
3. **Use environment variables** for production
4. **Rotate keys** if exposed

### Input Validation

1. **Sanitize user input** (sentence count)
2. **Validate API responses** before processing
3. **Check file paths** for directory traversal

### Error Messages

1. **Don't expose API keys** in logs
2. **Sanitize stack traces** in production
3. **Log errors** securely without sensitive data

---

## Deployment Checklist

- [ ] All files under 200 lines
- [ ] All functions have docstrings
- [ ] All complex logic has inline comments
- [ ] API key in .env (not hardcoded)
- [ ] .gitignore includes .env and results/
- [ ] README.md has setup instructions
- [ ] Tests pass with >70% coverage
- [ ] Pipeline runs end-to-end successfully
- [ ] Visualizations have correct colors and metadata
- [ ] All results saved to results/ folder

---

*Document Version: 1.0*
*Last Updated: 2025-11-04*
*Author: Claude (AI Assistant)*