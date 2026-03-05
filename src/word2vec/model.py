from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class Gradients(NamedTuple):
    loss: float
    grad_w_in: np.ndarray
    grad_w_out_ctx: np.ndarray
    grad_w_out_neg: np.ndarray


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


@dataclass
class SkipGramNS:
    vocab_size: int
    embedding_dim: int
    seed: int = 42

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        scale = 0.5 / self.embedding_dim
        self.w_in = rng.uniform(-scale, scale, (self.vocab_size, self.embedding_dim))
        self.w_out = np.zeros((self.vocab_size, self.embedding_dim))

    def forward(self, center_id: int, context_ids: np.ndarray) -> np.ndarray:
        v_center = self.w_in[center_id]
        v_contexts = self.w_out[context_ids]
        return sigmoid(v_contexts @ v_center)

    def compute_loss(self, center_id: int, context_id: int, negative_ids: np.ndarray) -> float:
        """Compute SGNS loss (used by gradient check tests)."""
        v_center = self.w_in[center_id]
        sig_pos = sigmoid(v_center @ self.w_out[context_id])
        sig_neg = sigmoid(self.w_out[negative_ids] @ v_center)
        return -np.log(sig_pos + 1e-10) - np.sum(np.log(1 - sig_neg + 1e-10))

    def compute_gradients(
        self, center_id: int, context_id: int, negative_ids: np.ndarray
    ) -> Gradients:
        """Compute loss and analytical gradients in a single forward pass."""
        v_center = self.w_in[center_id]
        v_context = self.w_out[context_id]
        v_negatives = self.w_out[negative_ids]

        sig_pos = sigmoid(v_center @ v_context)
        sig_neg = sigmoid(v_negatives @ v_center)

        loss = -np.log(sig_pos + 1e-10) - np.sum(np.log(1 - sig_neg + 1e-10))
        grad_w_in = (sig_pos - 1) * v_context + sig_neg @ v_negatives
        grad_w_out_ctx = (sig_pos - 1) * v_center
        grad_w_out_neg = sig_neg[:, np.newaxis] * v_center[np.newaxis, :]
        return Gradients(loss, grad_w_in, grad_w_out_ctx, grad_w_out_neg)

    def update(
        self,
        center_id: int,
        context_id: int,
        negative_ids: np.ndarray,
        grads: Gradients,
        lr: float,
    ) -> None:
        """SGD parameter update: W -= lr * gradient."""
        self.w_in[center_id] -= lr * grads.grad_w_in
        self.w_out[context_id] -= lr * grads.grad_w_out_ctx
        self.w_out[negative_ids] -= lr * grads.grad_w_out_neg

    def train_batch(
        self,
        center_ids: np.ndarray,
        context_ids: np.ndarray,
        negative_ids: np.ndarray,
        lr: float,
    ) -> float:
        """Vectorized forward + backward + update for a batch. Returns total batch loss."""
        # center_ids: (B,), context_ids: (B,), negative_ids: (B, K)
        v_centers = self.w_in[center_ids]  # (B, D)
        v_contexts = self.w_out[context_ids]  # (B, D)
        v_negatives = self.w_out[negative_ids]  # (B, K, D)

        # Forward — positive: element-wise multiply + sum along D
        pos_scores = np.sum(v_centers * v_contexts, axis=1)  # (B,)
        sig_pos = sigmoid(pos_scores)  # (B,)

        # Forward — negative: each center dot its K negatives
        neg_scores = np.einsum("bd,bkd->bk", v_centers, v_negatives)  # (B, K)
        sig_neg = sigmoid(neg_scores)  # (B, K)

        # Loss
        batch_loss = -np.sum(np.log(sig_pos + 1e-10)) - np.sum(np.log(1 - sig_neg + 1e-10))

        # Gradients
        sp1 = (sig_pos - 1.0)[:, np.newaxis]  # (B, 1)
        grad_w_in = sp1 * v_contexts + np.einsum("bk,bkd->bd", sig_neg, v_negatives)
        grad_w_out_ctx = sp1 * v_centers  # (B, D)
        grad_w_out_neg = sig_neg[:, :, np.newaxis] * v_centers[:, np.newaxis, :]

        # Update — use np.add.at for correct accumulation of duplicate indices
        np.add.at(self.w_in, center_ids, -lr * grad_w_in)
        np.add.at(self.w_out, context_ids, -lr * grad_w_out_ctx)
        np.add.at(self.w_out, negative_ids, -lr * grad_w_out_neg)

        return float(batch_loss)
