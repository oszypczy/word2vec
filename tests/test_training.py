import numpy as np

from word2vec.model import SkipGramNS
from word2vec.preprocessing import build_negative_sampling_table
from word2vec.training import train_epoch


def test_train_epoch_returns_avg_loss():
    """train_epoch should return average loss as a positive float."""
    model = SkipGramNS(vocab_size=10, embedding_dim=5, seed=0)
    freqs = np.ones(10, dtype=np.float64)
    neg_table = build_negative_sampling_table(freqs, table_size=1000)
    pairs = np.array([(0, 1), (2, 3), (4, 5)], dtype=np.int64)
    rng = np.random.default_rng(0)

    result = train_epoch(model, pairs, neg_table, lr=0.025, neg_samples=3, rng=rng)
    assert isinstance(result, float)
    assert result > 0


def test_train_epoch_loss_decreases():
    """Loss should decrease over consecutive epochs."""
    model = SkipGramNS(vocab_size=20, embedding_dim=10, seed=42)
    freqs = np.ones(20, dtype=np.float64)
    neg_table = build_negative_sampling_table(freqs, table_size=1000)

    # Repeated patterns so the model can actually learn something
    pairs = np.array([(i % 20, (i + 1) % 20) for i in range(200)], dtype=np.int64)
    rng = np.random.default_rng(42)

    loss1 = train_epoch(model, pairs, neg_table, lr=0.05, neg_samples=5, rng=rng)
    loss2 = train_epoch(model, pairs, neg_table, lr=0.05, neg_samples=5, rng=rng)

    assert loss2 < loss1, f"Loss should decrease: {loss1} -> {loss2}"
