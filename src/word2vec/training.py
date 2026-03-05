import numpy as np
from tqdm import tqdm

from word2vec.model import SkipGramNS


def train_epoch(
    model: SkipGramNS,
    training_pairs: np.ndarray,
    neg_table: np.ndarray,
    lr: float,
    neg_samples: int,
    rng: np.random.Generator,
    epoch_desc: str = "",
    batch_size: int = 256,
) -> float:
    """Train one epoch over all training pairs. Returns average loss."""
    total_loss = 0.0
    n = len(training_pairs)

    # Shuffle pairs
    order = rng.permutation(n)
    pairs_shuffled = training_pairs[order]

    # Pre-generate all negative samples for the epoch
    all_neg_indices = rng.integers(0, len(neg_table), size=(n, neg_samples))
    all_negatives = neg_table[all_neg_indices]

    num_batches = (n + batch_size - 1) // batch_size
    for b in tqdm(range(num_batches), desc=epoch_desc, unit="batch", leave=False):
        start = b * batch_size
        end = min(start + batch_size, n)

        centers = pairs_shuffled[start:end, 0]
        contexts = pairs_shuffled[start:end, 1]
        negatives = all_negatives[start:end]

        batch_loss = model.train_batch(centers, contexts, negatives, lr)
        total_loss += batch_loss

    return total_loss / n


def train(
    model: SkipGramNS,
    training_pairs: np.ndarray,
    neg_table: np.ndarray,
    epochs: int = 5,
    lr_start: float = 0.025,
    lr_min: float = 1e-4,
    neg_samples: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> list[float]:
    """Full training loop with linear LR decay. Returns loss history per epoch."""
    rng = np.random.default_rng(seed)
    loss_history = []

    for epoch in range(epochs):
        lr = lr_start - (lr_start - lr_min) * (epoch / max(epochs - 1, 1))
        desc = f"Epoch {epoch + 1}/{epochs} (lr={lr:.5f})"
        avg_loss = train_epoch(model, training_pairs, neg_table, lr, neg_samples, rng, desc)
        loss_history.append(avg_loss)

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} | LR: {lr:.6f} | Avg Loss: {avg_loss:.4f}")

    return loss_history
