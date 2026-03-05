import numpy as np

from word2vec.model import SkipGramNS


def test_init_shapes():
    model = SkipGramNS(vocab_size=100, embedding_dim=50)
    assert model.w_in.shape == (100, 50)
    assert model.w_out.shape == (100, 50)


def test_forward_output_range():
    """Sigmoid output should always be in (0, 1)."""
    model = SkipGramNS(vocab_size=100, embedding_dim=50, seed=42)
    scores = model.forward(center_id=0, context_ids=np.array([1, 2, 3]))
    assert scores.shape == (3,)
    assert np.all(scores > 0) and np.all(scores < 1)


def test_loss_positive():
    """Loss should be non-negative."""
    model = SkipGramNS(vocab_size=100, embedding_dim=50, seed=42)
    loss = model.compute_loss(center_id=5, context_id=10, negative_ids=np.array([20, 30, 40]))
    assert loss > 0


def test_gradient_check():
    """Numerical vs analytical gradient — the most important test."""
    model = SkipGramNS(vocab_size=20, embedding_dim=10, seed=42)

    center_id = 3
    context_id = 7
    negative_ids = np.array([1, 5, 12])

    # Analytical gradients
    loss, grad_in, grad_out_ctx, grad_out_neg = model.compute_gradients(
        center_id, context_id, negative_ids
    )

    epsilon = 1e-5

    # Check gradient for W_in[center_id]
    numerical_grad_in = np.zeros_like(grad_in)
    for i in range(len(grad_in)):
        model.w_in[center_id, i] += epsilon
        loss_plus = model.compute_loss(center_id, context_id, negative_ids)
        model.w_in[center_id, i] -= 2 * epsilon
        loss_minus = model.compute_loss(center_id, context_id, negative_ids)
        model.w_in[center_id, i] += epsilon  # restore
        numerical_grad_in[i] = (loss_plus - loss_minus) / (2 * epsilon)

    np.testing.assert_allclose(grad_in, numerical_grad_in, rtol=1e-4, atol=1e-6)

    # Check gradient for W_out[context_id]
    numerical_grad_out_ctx = np.zeros_like(grad_out_ctx)
    for i in range(len(grad_out_ctx)):
        model.w_out[context_id, i] += epsilon
        loss_plus = model.compute_loss(center_id, context_id, negative_ids)
        model.w_out[context_id, i] -= 2 * epsilon
        loss_minus = model.compute_loss(center_id, context_id, negative_ids)
        model.w_out[context_id, i] += epsilon
        numerical_grad_out_ctx[i] = (loss_plus - loss_minus) / (2 * epsilon)

    np.testing.assert_allclose(grad_out_ctx, numerical_grad_out_ctx, rtol=1e-4, atol=1e-6)

    # Check gradient for W_out[negative_ids[0]]
    neg_id = negative_ids[0]
    numerical_grad_out_neg0 = np.zeros(model.w_out.shape[1])
    for i in range(len(numerical_grad_out_neg0)):
        model.w_out[neg_id, i] += epsilon
        loss_plus = model.compute_loss(center_id, context_id, negative_ids)
        model.w_out[neg_id, i] -= 2 * epsilon
        loss_minus = model.compute_loss(center_id, context_id, negative_ids)
        model.w_out[neg_id, i] += epsilon
        numerical_grad_out_neg0[i] = (loss_plus - loss_minus) / (2 * epsilon)

    np.testing.assert_allclose(grad_out_neg[0], numerical_grad_out_neg0, rtol=1e-4, atol=1e-6)


def test_update_reduces_loss():
    """After one SGD step, loss for the same example should decrease."""
    model = SkipGramNS(vocab_size=20, embedding_dim=10, seed=42)
    center_id = 3
    context_id = 7
    negative_ids = np.array([1, 5, 12])

    grads = model.compute_gradients(center_id, context_id, negative_ids)
    loss_before = grads.loss

    model.update(center_id, context_id, negative_ids, grads, lr=0.1)

    loss_after = model.compute_loss(center_id, context_id, negative_ids)
    assert loss_after < loss_before, f"Loss should decrease: {loss_before} -> {loss_after}"
