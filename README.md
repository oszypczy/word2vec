# Word2Vec — Skip-Gram with Negative Sampling

A from-scratch implementation of the Word2Vec Skip-Gram model with Negative Sampling (SGNS) in pure NumPy, accelerated with Numba JIT compilation. Trained and evaluated on the Text8 dataset.

## Requirements

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) package manager

## Quick Start

```bash
# Install dependencies
uv sync

# Train on Text8 with default hyperparameters
uv run python -m word2vec

# Train with custom parameters
uv run python -m word2vec --dim 300 --epochs 10 --window 6 --neg-samples 13

# Run tests
uv run pytest
```

### Notebooks

Install notebook dependencies and launch Jupyter:

```bash
uv sync --group notebook
uv run jupyter notebook
```

## Project Structure

```
src/word2vec/
├── __main__.py       # CLI entry point (Typer)
├── preprocessing.py  # Tokenization, vocabulary, subsampling, training pairs
├── model.py          # SkipGramNS model with Numba-accelerated training
└── training.py       # Training loop with LR decay
tests/
├── test_preprocessing.py
├── test_model.py
└── test_training.py
notebooks/
├── hyperparameter_optimization.ipynb
├── evaluation_benchmark.ipynb
└── embedding_visualization.ipynb
```

## How It Works

### Preprocessing (`preprocessing.py`)

The preprocessing pipeline converts raw text into training data in five steps:

1. **Tokenization** — `tokenize()` splits text on whitespace and lowercases. No stemming or lemmatization — Word2Vec relies on raw surface forms.

2. **Vocabulary construction** — `build_vocab()` counts token frequencies, filters words below `min_count`, and builds a `Vocabulary` with word-to-id / id-to-word mappings. Words are sorted by frequency descending, so index 0 is the most frequent word. The `min_count` threshold removes noisy rare words that don't have enough context to learn good embeddings.

3. **Subsampling of frequent words** — `subsample_probs()` computes discard probabilities using the formula from the original paper: `P(discard) = 1 - sqrt(t / f)` where `f` is the word's relative frequency and `t` is a threshold (typically 1e-5). `apply_subsampling()` then randomly removes tokens based on these probabilities. This downsamples extremely frequent words like "the", "a", "is" which carry little semantic signal but dominate training pairs.

4. **Training pair generation** — `generate_training_pairs()` creates (center, context) pairs using a sliding window. For each offset from 1 to `window_size`, it generates pairs in both directions (left and right context) using vectorized array slicing — no Python loops over the corpus.

5. **Negative sampling table** — `build_negative_sampling_table()` builds a large lookup table (10M entries) where each word appears proportionally to `freq^0.75`. The 3/4 power smooths the distribution — it boosts rare words and dampens frequent ones compared to the raw unigram distribution. During training, random indices into this table give negative samples.

### Model (`model.py`)

`SkipGramNS` is the core model with two embedding matrices:

- **`w_in`** (vocab_size × embedding_dim) — input (center) embeddings, initialized with small uniform random values scaled by `0.5 / embedding_dim`
- **`w_out`** (vocab_size × embedding_dim) — output (context) embeddings, initialized to zeros

The **SGNS loss** for a (center, context) pair with K negative samples is:

```
L = -log σ(v_center · v_context) - Σ_k log σ(-v_center · v_negative_k)
```

where σ is the sigmoid function. The first term pushes center and context embeddings closer; the second term pushes center and negative embeddings apart.

**Training acceleration:** The hot loop (`_train_batch_jit`) is compiled with Numba's `@njit(parallel=True)`. It replaces NumPy operations (`einsum`, `add.at`) with explicit loops over the embedding dimensions, which Numba compiles to native SIMD code with zero temporary allocations. The `nb.prange` parallelizes across the batch dimension.

The model also provides `compute_gradients()` for analytical gradient computation (used in gradient check tests) and `save()`/`load()` for persistence via NumPy's `.npz` format.

### Training (`training.py`)

- `train_epoch()` — shuffles training pairs, pre-generates all negative samples for the epoch (single random call), then iterates in mini-batches calling `model.train_batch()`.
- `train()` — runs multiple epochs with linear learning rate decay from `lr_start` to `lr_min`.

## Notebooks

### 1. Hyperparameter Optimization (`hyperparameter_optimization.ipynb`)

Bayesian hyperparameter search using **Optuna** (TPE sampler) with 8 hyperparameters:

| Parameter | Search Space |
|-----------|-------------|
| `embedding_dim` | {50, 100, 200, 300} |
| `window` | 2–10 |
| `neg_samples` | 2–15 |
| `lr_start` | 0.005–0.1 (log) |
| `lr_min` | 1e-5–1e-3 (log) |
| `min_count` | 3–20 |
| `epochs` | 3–10 |
| `subsample_t` | 1e-6–1e-4 (log) |

**Metric:** Spearman ρ on WordSim-353 benchmark. **Pruning:** MedianPruner kills unpromising trials after each epoch.

**Key findings:**
- Best result: **ρ = 0.706** (trial #6: dim=300, window=6, neg=13, min_count=11, epochs=10)
- `min_count` is the most important hyperparameter (importance 0.25) — it controls vocabulary size and benchmark coverage
- `lr_start` has minimal impact (0.04) — model is robust to initial learning rate
- 5–6 epochs are sufficient; longer training shows diminishing returns

### 2. Evaluation Benchmark (`evaluation_benchmark.ipynb`)

Comprehensive comparison of **optimized vs default** hyperparameters on standard benchmarks:

**Word Similarity** (Spearman ρ):

| Benchmark | Optimized | Default |
|-----------|-----------|---------|
| WordSim-353 | 0.70 | 0.68 |
| SimLex-999 | 0.29 | 0.29 |
| MEN-3000 | 0.67 | 0.61 |

**Word Analogies** (Google dataset, ~19k questions):

| Category | Optimized | Default |
|----------|-----------|---------|
| Semantic | 36.1% | 16.1% |
| Syntactic | 26.4% | 17.0% |
| **Total** | **30.1%** | **16.7%** |

Optimization nearly doubled analogy accuracy, with the largest gains in semantic categories (capital-country, nationality-adjective).

### 3. Embedding Visualization (`embedding_visualization.ipynb`)

Visual exploration of the learned embedding space:

- **t-SNE of semantic clusters** — ~100 words from 6 categories (countries, capitals, numbers, animals, technology, colors) projected to 2D. Categories form clear, well-separated clusters.
- **t-SNE of top-500 words** — global embedding structure with semantic categories highlighted.
- **Countries & Capitals zoom** — pairs connected by dashed lines show that capitals cluster near their countries (france↔paris, germany↔berlin).
- **Cosine similarity heatmap** — hierarchically clustered matrix reveals semantic blocks. Numbers form the tightest cluster; countries pair with their capitals in the dendrogram.

## References

- Mikolov, T., et al. "Efficient Estimation of Word Representations in Vector Space." (2013)
- Mikolov, T., et al. "Distributed Representations of Words and Phrases and their Compositionality." (2013)
