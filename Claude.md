# Claude Implementation Guide
## K-Means & KNN Sentence Classification System

---

## Document Purpose

This document serves as Claude's working context and implementation tracker for building the K-Means and KNN sentence classification system. It contains:

1. **Project Context** - Quick reference for the project goals and architecture
2. **Implementation Status** - Current progress and completed tasks
3. **Active Work** - What Claude is currently implementing
4. **Implementation Notes** - Decisions, gotchas, and learnings
5. **Code Snippets** - Reusable code patterns
6. **Next Steps** - Immediate priorities

---

## Project Context

### High-Level Overview
Build an ML pipeline that:
1. Generates categorized sentences (animals/green, music/blue, food/red)
2. Converts sentences to vectors using **OpenAI embeddings** (`text-embedding-3-small`)
3. Applies **K-Means clustering** (unsupervised learning)
4. Applies **KNN classification** (supervised learning)
5. Visualizes results with **UMAP dimensionality reduction**
6. Saves all outputs to `results/` folder with metadata

### Technology Stack
- **Package Manager**: uv (fast Python package installer)
- **LLM**: OpenAI API for embeddings (1536 dimensions)
- **ML**: scikit-learn (K-Means, KNN)
- **Visualization**: matplotlib + UMAP
- **Utilities**: numpy, python-dotenv, tqdm

### Key Requirements
✅ **Vector Normalization**: All vectors MUST be L2 normalized before classification
✅ **Color Mapping**: Animals=Green, Music=Blue, Food=Red
✅ **Metadata**: Each visualization includes algorithm info, metrics, timestamp
✅ **Results Storage**: All outputs saved to `results/` folder with timestamps

### Project Structure
```
Ex16-KNN-AlgoPractice/
├── .env                          # OpenAI API key
├── .gitignore
├── pyproject.toml               # uv configuration
├── README.md
├── PRD.md                        # Product requirements
├── planning.md                   # Implementation plan
├── tasks.json                    # Task breakdown
├── Claude.md                     # THIS FILE
│
├── src/
│   ├── __init__.py
│   ├── main.py                   # Pipeline orchestration
│   ├── config.py                 # Configuration management
│   ├── sentence_generator.py    # Generate categorized sentences
│   ├── vectorization_agent.py   # OpenAI embedding agent
│   ├── clustering.py             # K-Means implementation
│   ├── classification.py         # KNN implementation
│   ├── visualization.py          # UMAP + matplotlib plotting
│   └── utils.py                  # Helper functions
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Pytest fixtures
│   ├── test_sentence_generator.py
│   ├── test_vectorization.py
│   ├── test_clustering.py
│   ├── test_classification.py
│   ├── test_utils.py
│   └── test_integration.py
│
└── results/                      # Git-ignored outputs
    ├── kmeans_clustering_*.png
    ├── knn_classification_*.png
    ├── training_data_*.json
    ├── test_data_*.json
    ├── vectors_*.npz
    ├── metrics_*.json
    └── run_log_*.txt
```

---

## Implementation Status

### Phase 1: Project Setup ⏳ IN PROGRESS
- [ ] Task 1: Initialize uv project (NOT STARTED)
- [ ] Task 2: Install dependencies with uv (NOT STARTED)
- [ ] Task 3: Create project folder structure (NOT STARTED)
- [ ] Task 4: Create .env and .gitignore files (NOT STARTED)
- [ ] Task 5: Create README.md (NOT STARTED)

### Phase 2: Core Infrastructure ⏸️ PENDING
- [ ] Task 6: Implement config.py
- [ ] Task 7: Create utils.py with helper functions
- [ ] Task 8: Set up logging system
- [ ] Task 9: Create base test structure

### Phase 3: Sentence Generation ⏸️ PENDING
- [ ] Task 10: Implement SentenceGenerator class
- [ ] Task 11: Create sentence templates
- [ ] Task 12: Add validation for sentence quality

### Phase 4: Vectorization Agent ⏸️ PENDING
- [ ] Task 13: Create VectorizationAgent class
- [ ] Task 14: Implement OpenAI API integration
- [ ] Task 15: Add batch processing with progress bar
- [ ] Task 16: Implement vector normalization and validation

### Phase 5: K-Means Clustering ⏸️ PENDING
- [ ] Task 17: Create KMeansClustering class
- [ ] Task 18: Implement clustering with scikit-learn
- [ ] Task 19: Calculate cluster metrics
- [ ] Task 20: Generate confusion matrix

### Phase 6: KNN Classification ⏸️ PENDING
- [ ] Task 21: Create KNNClassifier class
- [ ] Task 22: Train KNN on clustered data
- [ ] Task 23: Predict on test data
- [ ] Task 24: Calculate classification metrics

### Phase 7: Visualization ⏸️ PENDING
- [ ] Task 25: Implement dimensionality reduction (UMAP)
- [ ] Task 26: Create K-Means visualization
- [ ] Task 27: Create KNN visualization
- [ ] Task 28: Add legends and color mapping
- [ ] Task 29: Save visualizations to results/

### Phase 8: Main Pipeline Integration ⏸️ PENDING
- [ ] Task 30: Implement main.py orchestration
- [ ] Task 31: Add command-line argument parsing
- [ ] Task 32: Integrate all modules
- [ ] Task 33: Implement results storage system

### Phase 9: Testing & Validation ⏸️ PENDING
- [ ] Task 34: Write unit tests for sentence generation
- [ ] Task 35: Write unit tests for vectorization
- [ ] Task 36: Write unit tests for clustering and KNN
- [ ] Task 37: Write integration test
- [ ] Task 38: Add validation checks for normalization

### Phase 10: Documentation & Polish ⏸️ PENDING
- [ ] Task 39: Write comprehensive README.md
- [ ] Task 40: Add docstrings to all functions
- [ ] Task 41: Create example runs
- [ ] Task 42: Final code review and cleanup

---

## Active Work

### Current Task: NONE (Awaiting user confirmation to start)

**Next Immediate Steps**:
1. Initialize uv project (Task 1)
2. Install dependencies (Task 2)
3. Create folder structure (Task 3)
4. Set up .env and .gitignore (Task 4)
5. Write initial README (Task 5)

**Waiting On**:
- User confirmation to begin implementation
- OpenAI API key (user will provide in .env)

---

## Implementation Notes

### Critical Decisions

#### 1. OpenAI Embedding Model
**Decision**: Use `text-embedding-3-small`
**Rationale**:
- Cost-effective: $0.02 per 1M tokens (total project cost <$0.01)
- Fast: ~100 sentences/second
- 1536 dimensions (good quality, manageable size)
- Latest model from OpenAI

**Alternative Considered**: `text-embedding-3-large` (3072 dims, higher quality but overkill)

#### 2. Dimensionality Reduction
**Decision**: Use UMAP as primary method
**Rationale**:
- Preserves both local and global structure
- Faster than t-SNE for datasets >500 samples
- Better for visualization clarity
- Fallback options: t-SNE, PCA

#### 3. Sentence Generation Strategy
**Decision**: Template-based generation with randomization
**Rationale**:
- Deterministic and reproducible
- No additional API costs
- Fast generation
- Easy to ensure train/test separation

**Alternative Considered**: Use OpenAI to generate sentences (more variety but costs money)

#### 4. Vector Normalization
**Decision**: Apply L2 normalization immediately after embedding generation
**Rationale**:
- KNN with Euclidean distance benefits from normalization
- Prevents magnitude bias in distance calculations
- Ensures fair comparison between vectors

**Validation**: `assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)`

---

### Known Gotchas & Solutions

#### Gotcha 1: OpenAI API Rate Limits
**Problem**: Free tier has rate limits
**Solution**:
- Batch requests (up to 2048 sentences per call)
- Implement exponential backoff (1s, 2s, 4s)
- Add retry logic (max 3 attempts)

#### Gotcha 2: UMAP Randomness
**Problem**: UMAP produces different layouts each run
**Solution**: Set `random_state=42` for reproducibility

#### Gotcha 3: Cluster Label Mismatch
**Problem**: K-Means cluster IDs (0,1,2) don't match true categories
**Solution**: Use confusion matrix + optimal assignment (scipy.optimize.linear_sum_assignment)

#### Gotcha 4: Memory with Large Embeddings
**Problem**: 1536-dim vectors use significant RAM
**Solution**:
- Use `float32` instead of `float64`
- Process in batches
- Clear intermediate variables

#### Gotcha 5: Windows Path Issues
**Problem**: Windows uses backslashes in paths
**Solution**: Use `pathlib.Path` or `os.path.join` for cross-platform compatibility

---

## Code Snippets & Patterns

### 1. L2 Normalization (utils.py)
```python
import numpy as np

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize vectors to unit length using L2 norm.

    Args:
        vectors: Array of shape (n_samples, n_features)

    Returns:
        Normalized vectors of same shape
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return vectors / norms

def validate_normalization(vectors: np.ndarray, tolerance: float = 1e-5) -> None:
    """
    Validate that vectors are normalized (||v|| = 1).

    Raises:
        ValueError: If vectors are not normalized within tolerance
    """
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=tolerance):
        raise ValueError(f"Vectors not normalized! Norms range: {norms.min():.6f} to {norms.max():.6f}")
```

### 2. OpenAI API Call with Retry (vectorization_agent.py)
```python
import time
import openai
from typing import List
import numpy as np

def _call_openai_with_retry(self, sentences: List[str], max_attempts: int = 3) -> np.ndarray:
    """
    Call OpenAI API with exponential backoff retry logic.

    Args:
        sentences: List of sentences to embed
        max_attempts: Maximum retry attempts

    Returns:
        Embeddings as numpy array
    """
    for attempt in range(max_attempts):
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=sentences,
                encoding_format="float"
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings, dtype=np.float32)

        except openai.RateLimitError as e:
            if attempt < max_attempts - 1:
                delay = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                self.logger.warning(f"Rate limited. Retrying in {delay}s... (attempt {attempt+1}/{max_attempts})")
                time.sleep(delay)
            else:
                raise

        except openai.AuthenticationError as e:
            self.logger.error("Invalid API key. Check your .env file.")
            raise

        except openai.APIConnectionError as e:
            if attempt < max_attempts - 1:
                delay = 2 ** attempt
                self.logger.warning(f"Connection error. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise
```

### 3. Configuration Loading (config.py)
```python
import os
from dataclasses import dataclass
from typing import Dict
from dotenv import load_dotenv

@dataclass
class Config:
    """Project configuration with defaults."""

    # OpenAI settings
    openai_api_key: str
    openai_model: str = "text-embedding-3-small"
    openai_timeout: int = 30

    # Data generation
    sentences_per_category: int = 10
    categories: tuple = ("animals", "music", "food")

    # ML parameters
    kmeans_clusters: int = 3
    knn_neighbors: int = 5
    random_state: int = 42

    # Visualization
    colors: Dict[str, str] = None
    reduction_method: str = "umap"

    # Output
    results_folder: str = "results"

    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                "animals": "#00FF00",  # Green
                "music": "#0000FF",    # Blue
                "food": "#FF0000"      # Red
            }

    @classmethod
    def from_env(cls, env_path: str = ".env"):
        """Load configuration from environment variables."""
        load_dotenv(env_path)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")

        return cls(
            openai_api_key=api_key,
            openai_model=os.getenv("OPENAI_MODEL", "text-embedding-3-small"),
            openai_timeout=int(os.getenv("OPENAI_TIMEOUT", "30"))
        )
```

### 4. Timestamp Generation (utils.py)
```python
from datetime import datetime

def generate_timestamp() -> str:
    """Generate timestamp string for filenames (YYYYMMDD_HHMMSS)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
```

### 5. Progress Bar with tqdm (vectorization_agent.py)
```python
from tqdm import tqdm

def vectorize_batch(self, sentences: List[str], batch_size: int = 100) -> np.ndarray:
    """Vectorize sentences in batches with progress bar."""
    all_embeddings = []

    for i in tqdm(range(0, len(sentences), batch_size), desc="Vectorizing"):
        batch = sentences[i:i + batch_size]
        embeddings = self._call_openai_with_retry(batch)
        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)
```

### 6. UMAP Dimensionality Reduction (visualization.py)
```python
from umap import UMAP

def reduce_dimensions(vectors: np.ndarray, method: str = "umap", random_state: int = 42) -> np.ndarray:
    """
    Reduce high-dimensional vectors to 2D for visualization.

    Args:
        vectors: Array of shape (n_samples, n_features)
        method: Reduction method ('umap', 'tsne', or 'pca')
        random_state: Random seed for reproducibility

    Returns:
        2D array of shape (n_samples, 2)
    """
    if method == "umap":
        reducer = UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=random_state
        )
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=30)
    elif method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=random_state)
    else:
        raise ValueError(f"Unknown reduction method: {method}")

    return reducer.fit_transform(vectors)
```

### 7. Visualization with Metadata (visualization.py)
```python
import matplotlib.pyplot as plt
from datetime import datetime

def create_kmeans_visualization(
    vectors_2d: np.ndarray,
    labels: np.ndarray,
    colors: Dict[str, str],
    metrics: Dict,
    output_path: str
):
    """Create K-Means visualization with metadata."""
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # Scatter plot
    categories = ["animals", "music", "food"]
    for i, category in enumerate(categories):
        mask = labels == i
        ax.scatter(
            vectors_2d[mask, 0],
            vectors_2d[mask, 1],
            c=colors[category],
            label=f"{category.capitalize()} - {mask.sum()} samples",
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )

    # Title and labels
    ax.set_title("K-Means Clustering Results (k=3)", fontsize=16, fontweight='bold')
    ax.set_xlabel("UMAP Dimension 1", fontsize=12)
    ax.set_ylabel("UMAP Dimension 2", fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Metadata box
    metadata_text = f"""Algorithm: K-Means
Clusters: 3
Samples: {len(vectors_2d)}
Silhouette: {metrics['silhouette']:.3f}
Inertia: {metrics['inertia']:.2f}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"""

    ax.text(
        0.02, 0.98, metadata_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
```

---

## Testing Strategy

### Unit Tests
- **Mock OpenAI API**: Use `unittest.mock` or `pytest-mock` to avoid API calls
- **Fixtures**: Create sample vectors, sentences in `conftest.py`
- **Coverage Goal**: >70% code coverage

### Integration Test
- **Small Dataset**: Use 5 sentences per category for speed
- **End-to-End**: Run full pipeline and verify outputs
- **File Checks**: Ensure all expected files created in results/

### Test Commands
```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Run specific test
uv run pytest tests/test_vectorization.py -v
```

---

## Common Commands

### Development
```bash
# Initialize project
uv init

# Install dependency
uv add package-name

# Install dev dependency
uv add --dev package-name

# Run Python script
uv run python src/main.py

# Run with arguments
uv run python src/main.py --sentences 15 --k-neighbors 7

# Format code
uv run black src/ tests/

# Lint code
uv run ruff check src/ tests/

# Type check
uv run mypy src/ --ignore-missing-imports
```

### Git Workflow
```bash
# Initial commit
git init
git add .
git commit -m "Initial project setup"

# Feature branch
git checkout -b feature/vectorization
git add src/vectorization_agent.py
git commit -m "Implement OpenAI vectorization agent"
git checkout main
git merge feature/vectorization
```

---

## Environment Variables (.env)

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-your-api-key-here

# Optional: Model override
OPENAI_MODEL=text-embedding-3-small

# Optional: Timeout (seconds)
OPENAI_TIMEOUT=30

# Optional: Development mode
DEBUG=false
```

---

## Troubleshooting Guide

### Issue 1: "OPENAI_API_KEY not found"
**Solution**:
1. Check `.env` file exists in project root
2. Verify key is set: `OPENAI_API_KEY=sk-...`
3. No spaces around `=`
4. No quotes around value

### Issue 2: "Rate limit exceeded"
**Solution**:
- Wait 60 seconds and retry
- Reduce batch size
- Check OpenAI dashboard for quota

### Issue 3: "Import errors"
**Solution**:
```bash
# Reinstall dependencies
uv sync
# Or reinstall specific package
uv add --force openai
```

### Issue 4: "Vectors not normalized"
**Solution**:
- Check `normalize_vectors()` is called after embedding
- Verify tolerance: `atol=1e-5` might be too strict (try `1e-4`)

### Issue 5: "Results folder not found"
**Solution**:
- Add to `utils.py`: `os.makedirs("results", exist_ok=True)`

---

## Performance Benchmarks

### Expected Performance (on modern laptop)
- **Sentence Generation**: <1 second for 30 sentences
- **Vectorization**: ~3-5 seconds for 30 sentences (API dependent)
- **K-Means**: <1 second for 30 vectors
- **KNN**: <1 second for 30 vectors
- **Visualization**: ~2-3 seconds for UMAP + plotting
- **Total Pipeline**: ~10-15 seconds for 30 total sentences (10 per category)

### Scalability
- **Small** (5 per category): ~5 seconds total
- **Medium** (20 per category): ~20 seconds total
- **Large** (100 per category): ~60-90 seconds total

---

## API Cost Estimation

### OpenAI Pricing
- **Model**: text-embedding-3-small
- **Cost**: $0.020 per 1M tokens
- **Avg tokens per sentence**: ~15 tokens

### Project Costs
| Sentences per Category | Total Sentences | Total Tokens | Cost |
|------------------------|-----------------|--------------|------|
| 10 | 60 (30 train + 30 test) | ~900 | $0.000018 |
| 20 | 120 | ~1,800 | $0.000036 |
| 50 | 300 | ~4,500 | $0.00009 |
| 100 | 600 | ~9,000 | $0.00018 |

**Total estimated cost for development/testing**: **< $0.01**

---

## Quality Checklist

Before marking implementation complete, verify:

### Functionality
- [ ] Pipeline runs end-to-end without errors
- [ ] All 3 categories generate correctly
- [ ] Vectors are normalized (validation passes)
- [ ] K-Means produces 3 clusters
- [ ] KNN classifies test data
- [ ] Visualizations created successfully
- [ ] All files saved to results/ folder

### Code Quality
- [ ] All functions have docstrings
- [ ] Type hints added to function signatures
- [ ] No debug print statements left
- [ ] Code formatted with black
- [ ] Linting passes (ruff)
- [ ] No unused imports

### Testing
- [ ] Unit tests pass (>70% coverage)
- [ ] Integration test passes
- [ ] Edge cases handled (empty input, API errors)

### Documentation
- [ ] README has clear setup instructions
- [ ] Usage examples work
- [ ] .env.example provided
- [ ] Comments explain complex logic

### Results Validation
- [ ] Visualizations have correct colors (green/blue/red)
- [ ] Legends show category names and counts
- [ ] Metadata boxes display algorithm info
- [ ] Timestamps in filenames are correct format
- [ ] Metrics are reasonable (accuracy >80%, silhouette >0.5)

---

## Next Steps for Claude

### Immediate Actions (Phase 1)
1. **Task 1**: Initialize uv project
   - Run `uv init`
   - Create `pyproject.toml`

2. **Task 2**: Install dependencies
   - Add all required packages with uv

3. **Task 3**: Create folder structure
   - `mkdir src tests results`
   - Create `__init__.py` files

4. **Task 4**: Set up .env and .gitignore
   - Create .env template
   - Add comprehensive .gitignore

5. **Task 5**: Write README.md
   - Installation instructions
   - Usage examples

### After Phase 1 Completion
- Move to Phase 2: Core Infrastructure
- Start with config.py (Task 6)
- Then utils.py (Task 7)

---

## Claude's Working Notes

### Session Log

**Session 1 - 2025-11-04**
- Created PRD.md with comprehensive product requirements
- Created planning.md with 10-phase implementation plan
- Created tasks.json with 42 detailed tasks
- Created Claude.md (this file) for implementation tracking
- **Status**: Ready to begin Phase 1 implementation
- **Waiting on**: User confirmation to start coding

---

## References & Resources

### Documentation
- [uv Docs](https://docs.astral.sh/uv/)
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)

### Related Projects
- Sentence Transformers: https://www.sbert.net/
- UMAP Examples: https://umap-learn.readthedocs.io/en/latest/clustering.html

### Helpful Commands
```bash
# Check OpenAI usage
curl https://api.openai.com/v1/usage \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Test API key
uv run python -c "from openai import OpenAI; client = OpenAI(); print('API key valid!')"
```

---

## File Size Expectations

### Source Code
- Total Python code: ~1,500-2,000 lines
- Largest files:
  - `visualization.py`: ~400 lines
  - `vectorization_agent.py`: ~300 lines
  - `main.py`: ~300 lines

### Dependencies
- Virtual environment: ~500 MB
- Model downloads: None (uses API)

### Output Files
- Each PNG: ~200-500 KB
- Each JSON: ~10-50 KB
- Each NPZ: ~50-200 KB (depending on sentence count)
- Total per run: ~1-2 MB

---

## Completion Criteria

### Definition of Done
The project is complete when:
1. ✅ All 42 tasks in tasks.json are marked complete
2. ✅ Pipeline runs successfully from command line
3. ✅ All tests pass (>70% coverage)
4. ✅ Documentation is comprehensive and accurate
5. ✅ Code is clean, formatted, and linted
6. ✅ Example outputs demonstrate correct functionality
7. ✅ User can reproduce results following README

### Delivery Artifacts
1. Complete source code in `src/`
2. Comprehensive test suite in `tests/`
3. Documentation (README.md, PRD.md, planning.md, tasks.json, Claude.md)
4. Configuration files (.env.example, pyproject.toml, .gitignore)
5. Example outputs in `results/` (or screenshots)

---

*Last Updated: 2025-11-04*
*Claude Session: Initial setup and planning*
*Ready for implementation: Phase 1*