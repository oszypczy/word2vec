from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numba as nb
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


@nb.njit(cache=True, fastmath=True)
def _sigmoid_scalar(x: float) -> float:
    """Numerically stable sigmoid for a single scalar."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    ex = np.exp(x)
    return ex / (1.0 + ex)


@nb.njit(cache=True, fastmath=True, parallel=True)
def _train_batch_jit(
    w_in: np.ndarray,
    w_out: np.ndarray,
    center_ids: np.ndarray,
    context_ids: np.ndarray,
    negative_ids: np.ndarray,
    lr: float,
) -> float:
    """JIT-compiled train_batch: forward + backward + SGD update.

    Replaces np.einsum and np.add.at with explicit loops that Numba compiles
    to efficient native code with no temporary array allocations.
    """
    batch_size = center_ids.shape[0]
    neg_count = negative_ids.shape[1]
    dim = w_in.shape[1]
    total_loss = 0.0

    for i in nb.prange(batch_size):
        c_id = center_ids[i]
        ctx_id = context_ids[i]

        # --- forward: positive score ---
        pos_dot = 0.0
        for d in range(dim):
            pos_dot += w_in[c_id, d] * w_out[ctx_id, d]
        sig_pos = _sigmoid_scalar(pos_dot)

        # --- forward: negative scores + loss ---
        loss_i = -np.log(sig_pos + 1e-7)

        # Accumulate gradient for w_in[c_id] in a local buffer
        grad_center = np.empty(dim, dtype=np.float32)
        for d in range(dim):
            grad_center[d] = (sig_pos - 1.0) * w_out[ctx_id, d]

        for k in range(neg_count):
            n_id = negative_ids[i, k]
            neg_dot = 0.0
            for d in range(dim):
                neg_dot += w_in[c_id, d] * w_out[n_id, d]
            sig_neg = _sigmoid_scalar(neg_dot)
            loss_i -= np.log(1.0 - sig_neg + 1e-7)

            # Accumulate grad for center from this negative
            for d in range(dim):
                grad_center[d] += sig_neg * w_out[n_id, d]

            # Update w_out[n_id] (negative output embedding)
            for d in range(dim):
                w_out[n_id, d] -= lr * sig_neg * w_in[c_id, d]

        # Update w_out[ctx_id] (positive context embedding)
        for d in range(dim):
            w_out[ctx_id, d] -= lr * (sig_pos - 1.0) * w_in[c_id, d]

        # Update w_in[c_id] (center embedding)
        for d in range(dim):
            w_in[c_id, d] -= lr * grad_center[d]

        total_loss += loss_i

    return total_loss


@dataclass
class SkipGramNS:
    vocab_size: int
    embedding_dim: int
    seed: int = 42

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        scale = 0.5 / self.embedding_dim
        self.w_in = rng.uniform(-scale, scale, (self.vocab_size, self.embedding_dim)).astype(
            np.float32
        )
        self.w_out = np.zeros((self.vocab_size, self.embedding_dim), dtype=np.float32)

    def forward(self, center_id: int, context_ids: np.ndarray) -> np.ndarray:
        v_center = self.w_in[center_id]
        v_contexts = self.w_out[context_ids]
        return sigmoid(v_contexts @ v_center)

    def compute_loss(self, center_id: int, context_id: int, negative_ids: np.ndarray) -> float:
        """Compute SGNS loss (used by gradient check tests)."""
        v_center = self.w_in[center_id]
        sig_pos = sigmoid(v_center @ self.w_out[context_id])
        sig_neg = sigmoid(self.w_out[negative_ids] @ v_center)
        return -np.log(sig_pos + 1e-7) - np.sum(np.log(1 - sig_neg + 1e-7))

    def compute_gradients(
        self, center_id: int, context_id: int, negative_ids: np.ndarray
    ) -> Gradients:
        """Compute loss and analytical gradients in a single forward pass."""
        v_center = self.w_in[center_id]
        v_context = self.w_out[context_id]
        v_negatives = self.w_out[negative_ids]

        sig_pos = sigmoid(v_center @ v_context)
        sig_neg = sigmoid(v_negatives @ v_center)

        loss = -np.log(sig_pos + 1e-7) - np.sum(np.log(1 - sig_neg + 1e-7))
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
        """JIT-accelerated forward + backward + update for a batch. Returns total batch loss."""
        return _train_batch_jit(
            self.w_in, self.w_out, center_ids, context_ids, negative_ids, lr
        )

    def save(self, path: str | Path) -> None:
        """Save model weights to a .npz file."""
        np.savez(
            path,
            w_in=self.w_in,
            w_out=self.w_out,
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SkipGramNS":
        """Load model weights from a .npz file."""
        data = np.load(path)
        model = cls(
            vocab_size=int(data["vocab_size"]),
            embedding_dim=int(data["embedding_dim"]),
        )
        model.w_in = data["w_in"]
        model.w_out = data["w_out"]
        return model
