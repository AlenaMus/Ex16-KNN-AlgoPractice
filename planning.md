# Project Implementation Planning
## K-Means & KNN Sentence Classification System

**Project**: Sentence Embedding Classification with Visualization
**Tools**: Python, uv, OpenAI Embeddings, scikit-learn, matplotlib
**Date**: November 4, 2025

---

## Project Overview

This project implements a machine learning pipeline that:
1. Generates categorized sentences (animals, music, food)
2. Converts sentences to vectors using OpenAI embeddings
3. Applies K-Means clustering for unsupervised learning
4. Applies KNN classification for supervised learning
5. Visualizes results with color-coded scatter plots

---

## Technology Stack

### Package Manager
- **uv** - Fast Python package installer and resolver
- Benefits: 10-100x faster than pip, built-in virtual env management

### Core Dependencies
```toml
[project]
dependencies = [
    "openai>=1.12.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.4.0",
    "matplotlib>=3.8.0",
    "python-dotenv>=1.0.0",
    "umap-learn>=0.5.5",
    "tqdm>=4.66.0",
]
```

### OpenAI Model
- **Model**: `text-embedding-3-small`
- **Dimensions**: 1536
- **Cost**: ~$0.02 per 1M tokens (total project cost: <$0.01)
- **Speed**: ~100 sentences/second

---

## Project Structure

```
Ex16-KNN-AlgoPractice/
├── .env                          # OpenAI API key (not in git)
├── .gitignore                    # Ignore .env, results/, __pycache__
├── pyproject.toml               # uv project configuration
├── README.md                     # Project documentation
├── PRD.md                        # Product Requirements Document
├── planning.md                   # This file
├── tasks.json                    # Task breakdown for tracking
│
├── src/
│   ├── __init__.py
│   ├── main.py                   # Main entry point
│   ├── config.py                 # Configuration management
│   ├── sentence_generator.py    # Generate categorized sentences
│   ├── vectorization_agent.py   # OpenAI embedding agent
│   ├── clustering.py             # K-Means implementation
│   ├── classification.py         # KNN implementation
│   ├── visualization.py          # Plotting and visualization
│   └── utils.py                  # Helper functions
│
├── tests/
│   ├── __init__.py
│   ├── test_sentence_generator.py
│   ├── test_vectorization.py
│   ├── test_clustering.py
│   └── test_classification.py
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

---

## Implementation Phases

### Phase 1: Project Setup (Tasks 1-5)
**Goal**: Set up development environment and project structure

**Tasks**:
1. Initialize uv project with pyproject.toml
2. Create .env file for API key management
3. Set up folder structure (src/, tests/, results/)
4. Create .gitignore
5. Write README.md with setup instructions

**Deliverables**:
- Working uv environment
- Project skeleton ready for development

**Time Estimate**: 30 minutes

---

### Phase 2: Core Infrastructure (Tasks 6-9)
**Goal**: Build foundational modules

**Tasks**:
6. Implement config.py for settings management
7. Create utils.py with helper functions (normalization, validation)
8. Set up logging system
9. Create base test structure

**Key Components**:
- Configuration loader (from .env and defaults)
- Vector normalization function with validation
- Timestamp generation for file naming
- Error handling utilities

**Time Estimate**: 1 hour

---

### Phase 3: Sentence Generation (Tasks 10-12)
**Goal**: Generate categorized sentences

**Tasks**:
10. Implement SentenceGenerator class
11. Create sentence templates for 3 categories
12. Add validation for sentence quality

**Categories**:
- **Animals** (Green): Wildlife, pets, behavior
- **Music** (Blue): Instruments, genres, performance
- **Food** (Red): Cuisine, ingredients, cooking

**Approach**: Use rule-based generation with templates OR call OpenAI API for generation

**Time Estimate**: 1.5 hours

---

### Phase 4: Vectorization Agent (Tasks 13-16)
**Goal**: Implement OpenAI embedding integration

**Tasks**:
13. Create VectorizationAgent class
14. Implement OpenAI API integration with error handling
15. Add batch processing for efficiency
16. Implement vector normalization and validation

**Key Features**:
- Retry logic for API failures (3 attempts)
- Progress bar for batch processing
- L2 normalization verification
- Caching option for repeated runs

**API Configuration**:
```python
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=sentences,  # Batch up to 2048 items
    encoding_format="float"
)
```

**Time Estimate**: 2 hours

---

### Phase 5: K-Means Clustering (Tasks 17-20)
**Goal**: Implement unsupervised clustering

**Tasks**:
17. Create KMeansClustering class
18. Implement clustering with scikit-learn
19. Calculate cluster metrics (silhouette, inertia)
20. Generate confusion matrix vs true labels

**Configuration**:
```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=3,
    init='k-means++',
    max_iter=300,
    n_init=10,
    random_state=42
)
```

**Metrics to Track**:
- Silhouette score (cluster quality)
- Inertia (within-cluster sum of squares)
- Purity score vs true labels

**Time Estimate**: 1.5 hours

---

### Phase 6: KNN Classification (Tasks 21-24)
**Goal**: Implement supervised classification

**Tasks**:
21. Create KNNClassifier class
22. Train KNN on clustered data
23. Predict on test data
24. Calculate classification metrics

**Configuration**:
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    metric='euclidean',
    algorithm='auto'
)
```

**Metrics to Track**:
- Accuracy, Precision, Recall, F1-score (per class)
- Confusion matrix
- Confidence scores

**Time Estimate**: 1.5 hours

---

### Phase 7: Visualization (Tasks 25-29)
**Goal**: Create publication-quality visualizations

**Tasks**:
25. Implement dimensionality reduction (UMAP)
26. Create K-Means visualization with metadata
27. Create KNN visualization with metadata
28. Add legends and color mapping
29. Save visualizations to results/ folder

**Visualization Requirements**:
- **Dimensionality Reduction**: UMAP (2D)
- **Colors**: Animals=Green, Music=Blue, Food=Red
- **Size**: 12x8 inches, 300 DPI
- **Metadata Box**: Algorithm, parameters, accuracy, timestamp
- **Legend**: Category + color + sample count

**Layout Example**:
```
┌─────────────────────────────────────────────┐
│  K-Means Clustering Results (k=3)           │
├─────────────────────────────────────────────┤
│                                             │
│         [Scatter Plot with Colors]          │
│                                             │
│  Legend:                  Metadata:         │
│  ● Animals (Green) - 10   Algorithm: K-Means│
│  ● Music (Blue) - 10      Clusters: 3       │
│  ● Food (Red) - 10        Silhouette: 0.85  │
│                           Date: 2025-11-04  │
└─────────────────────────────────────────────┘
```

**Time Estimate**: 2 hours

---

### Phase 8: Main Pipeline Integration (Tasks 30-33)
**Goal**: Connect all components into cohesive pipeline

**Tasks**:
30. Implement main.py orchestration
31. Add command-line argument parsing
32. Integrate all modules (generation → vectorization → clustering → KNN → visualization)
33. Implement results storage system

**CLI Interface**:
```bash
# Basic usage
uv run python src/main.py --sentences 10

# Advanced usage
uv run python src/main.py --sentences 20 --k-neighbors 7 --reduction umap --no-cache
```

**Arguments**:
- `--sentences`: Number of sentences per category (default: 10)
- `--k-neighbors`: K value for KNN (default: 5)
- `--reduction`: Method for dimensionality reduction (umap/tsne/pca)
- `--cache`: Use cached vectors if available
- `--verbose`: Show detailed logging

**Pipeline Flow**:
```
1. Load config & validate API key
2. Generate training sentences (3n)
3. Generate test sentences (3n)
4. Vectorize training sentences → normalize
5. Vectorize test sentences → normalize
6. Run K-Means clustering
7. Visualize K-Means results
8. Train KNN on clusters
9. Predict test data with KNN
10. Visualize KNN results
11. Save all metrics and data
12. Print summary report
```

**Time Estimate**: 2 hours

---

### Phase 9: Testing & Validation (Tasks 34-38)
**Goal**: Ensure code quality and correctness

**Tasks**:
34. Write unit tests for sentence generation
35. Write unit tests for vectorization (mock OpenAI)
36. Write unit tests for clustering and KNN
37. Write integration test for full pipeline
38. Add validation checks for normalization

**Test Coverage Goals**:
- Unit tests: >70% coverage
- Integration test: Full happy path
- Edge cases: Empty input, API failures, invalid vectors

**Time Estimate**: 2 hours

---

### Phase 10: Documentation & Polish (Tasks 39-42)
**Goal**: Finalize project for delivery

**Tasks**:
39. Write comprehensive README.md
40. Add docstrings to all functions
41. Create example runs with different parameters
42. Final code review and cleanup

**README Sections**:
1. Project Overview
2. Installation (uv setup)
3. Configuration (.env setup)
4. Usage Examples
5. Output Explanation
6. Troubleshooting
7. Technical Details

**Time Estimate**: 1.5 hours

---

## Development Workflow

### 1. Environment Setup
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Initialize project
cd Ex16-KNN-AlgoPractice
uv init

# Install dependencies
uv add openai numpy scikit-learn matplotlib python-dotenv umap-learn tqdm

# Create .env file
echo "OPENAI_API_KEY=your-key-here" > .env
```

### 2. Development Commands
```bash
# Run main pipeline
uv run python src/main.py --sentences 10

# Run tests
uv run pytest tests/ -v

# Check code quality
uv run ruff check src/
uv run mypy src/

# Format code
uv run black src/
```

### 3. Git Workflow
```bash
# Initial commit
git init
git add .
git commit -m "Initial project setup"

# Feature branches
git checkout -b feature/sentence-generation
# ... develop ...
git commit -m "Implement sentence generation"
git checkout main
git merge feature/sentence-generation
```

---

## Critical Requirements Checklist

### Vector Normalization ✓
- [ ] All vectors normalized using L2 norm
- [ ] Validation check: `np.allclose(np.linalg.norm(vectors, axis=1), 1.0)`
- [ ] Applied to BOTH training and test vectors
- [ ] Verification before clustering and classification

### Color Mapping ✓
- [ ] Animals → Green (#00FF00)
- [ ] Music → Blue (#0000FF)
- [ ] Food → Red (#FF0000)
- [ ] Colors applied in both visualizations
- [ ] Legend shows color associations

### Visualization Metadata ✓
- [ ] Algorithm name displayed
- [ ] Parameters shown (k=3, n_neighbors=5)
- [ ] Timestamp included
- [ ] Sample counts per category
- [ ] Accuracy/metrics displayed

### Results Storage ✓
- [ ] All files saved to results/ folder
- [ ] Timestamped filenames (YYYYMMDD_HHMMSS)
- [ ] PNG visualizations at 300 DPI
- [ ] JSON data files for sentences and metrics
- [ ] NPZ file for vectors
- [ ] Log file for execution details

---

## Risk Mitigation

### API Rate Limits
**Risk**: OpenAI API rate limiting for free tier
**Mitigation**:
- Batch requests (up to 2048 sentences per call)
- Implement exponential backoff
- Cache vectors to avoid re-processing

### Poor Clustering Quality
**Risk**: Categories don't separate well in vector space
**Mitigation**:
- Use high-quality, distinct sentences
- Verify embeddings with cosine similarity checks
- Try different dimensionality reduction methods

### Memory Issues
**Risk**: Large vector arrays consume too much RAM
**Mitigation**:
- Use float32 instead of float64
- Process in batches if >1000 sentences
- Clear intermediate variables

---

## Success Metrics

### Quantitative
- ✓ K-Means silhouette score > 0.5
- ✓ KNN classification accuracy > 85%
- ✓ Pipeline execution time < 2 minutes (for 30 sentences)
- ✓ Test coverage > 70%

### Qualitative
- ✓ Clear visual separation in scatter plots
- ✓ Intuitive color mapping
- ✓ Easy-to-read documentation
- ✓ Reproducible results

---

## Timeline Estimate

| Phase | Duration | Cumulative |
|-------|----------|------------|
| 1. Project Setup | 0.5h | 0.5h |
| 2. Core Infrastructure | 1h | 1.5h |
| 3. Sentence Generation | 1.5h | 3h |
| 4. Vectorization Agent | 2h | 5h |
| 5. K-Means Clustering | 1.5h | 6.5h |
| 6. KNN Classification | 1.5h | 8h |
| 7. Visualization | 2h | 10h |
| 8. Main Pipeline | 2h | 12h |
| 9. Testing | 2h | 14h |
| 10. Documentation | 1.5h | 15.5h |

**Total Estimated Time**: 15-16 hours (2 full working days)

---

## Next Steps

1. ✅ Review this planning document
2. ✅ Review tasks.json for detailed task breakdown
3. ⏭️ Set up uv environment and install dependencies
4. ⏭️ Create OpenAI API key and add to .env
5. ⏭️ Start with Phase 1 tasks (project setup)

---

## Resources

### Documentation
- [uv Documentation](https://docs.astral.sh/uv/)
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)
- [scikit-learn K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [scikit-learn KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)

### Example Code Snippets
See tasks.json for detailed implementation examples for each task.

---

*Last Updated: November 4, 2025*