"""Text preprocessing for Word2Vec: tokenization, vocabulary, subsampling, training pairs."""

from collections import Counter
from typing import NamedTuple

import numpy as np


class Vocabulary(NamedTuple):
    """Vocabulary data structure holding words, mappings, and frequencies."""

    words: list[str]
    word2id: dict[str, int]
    id2word: dict[int, str]
    freqs: np.ndarray


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens."""
    return [word.lower() for word in text.split()]


def build_vocab(tokens: list[str], min_count: int = 5) -> Vocabulary:
    """Build vocabulary sorted by frequency descending, filtering by min_count."""
    filtered = {word: count for word, count in Counter(tokens).items() if count >= min_count}
    vocab = sorted(filtered.keys(), key=lambda w: filtered[w], reverse=True)
    word2id = {word: i for i, word in enumerate(vocab)}
    id2word = dict(enumerate(vocab))
    freqs = np.array([filtered[word] for word in vocab], dtype=np.float64)
    return Vocabulary(vocab, word2id, id2word, freqs)


def subsample_probs(freqs: np.ndarray, t: float = 1e-5) -> np.ndarray:
    """Compute discard probabilities for frequent words: P(discard) = 1 - sqrt(t / f)."""
    f = freqs / freqs.sum()
    probs = 1.0 - np.sqrt(t / f)
    return np.clip(probs, 0.0, 1.0)


def apply_subsampling(
    token_ids: np.ndarray, discard_probs: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Remove tokens randomly based on their discard probabilities."""
    keep_mask = rng.random(len(token_ids)) >= discard_probs[token_ids]
    return token_ids[keep_mask]


def generate_training_pairs(corpus: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Generate (center, context) pairs using vectorized offset slicing. Returns (N, 2) array."""
    all_centers = []
    all_contexts = []
    n = len(corpus)

    for offset in range(1, window_size + 1):
        # Positive offset: context is to the right of center
        all_centers.append(corpus[: n - offset])
        all_contexts.append(corpus[offset:])
        # Negative offset: context is to the left of center
        all_centers.append(corpus[offset:])
        all_contexts.append(corpus[: n - offset])

    centers = np.concatenate(all_centers)
    contexts = np.concatenate(all_contexts)
    return np.column_stack((centers, contexts))


def build_negative_sampling_table(freqs: np.ndarray, table_size: int = 10_000_000) -> np.ndarray:
    powered = freqs**0.75
    probs = powered / powered.sum()
    counts = (probs * table_size).astype(np.int64)
    return np.repeat(np.arange(len(freqs)), counts)
