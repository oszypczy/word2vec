import numpy as np

from word2vec.preprocessing import (
    build_vocab,
    generate_training_pairs,
    subsample_probs,
    tokenize,
)


def test_tokenize():
    text = "The Cat sat on The Mat the cat"
    tokens = tokenize(text)
    # Should lowercase and split on whitespace
    assert tokens == ["the", "cat", "sat", "on", "the", "mat", "the", "cat"]


def test_build_vocab_basic():
    tokens = ["the", "cat", "sat", "on", "the", "mat", "the", "cat"]
    vocab, word2id, id2word, freqs = build_vocab(tokens, min_count=1)
    # "the" appears 3x, should be in vocab
    assert "the" in word2id
    assert id2word[word2id["the"]] == "the"
    assert freqs[word2id["the"]] == 3
    # All 5 unique words with count >= 1: the, cat, sat, on, mat
    assert len(vocab) == 5


def test_build_vocab_min_count():
    tokens = ["the", "cat", "sat", "on", "the", "mat", "the", "cat"]
    vocab, word2id, id2word, freqs = build_vocab(tokens, min_count=2)
    assert "the" in word2id
    assert "cat" in word2id
    # "sat", "on", "mat" each appear once — below min_count=2
    assert "sat" not in word2id
    assert "on" not in word2id


def test_subsample_probs():
    freqs = np.array([100, 50, 10, 1], dtype=np.float64)
    probs = subsample_probs(freqs, t=1e-2)
    # More frequent words should have higher discard probability
    assert probs[0] > probs[1] > probs[2]
    # Rare word should have ~0 discard probability
    assert probs[3] < 0.01


def test_generate_training_pairs():
    corpus = np.array([0, 1, 2, 3, 4])
    pairs = generate_training_pairs(corpus, window_size=2)
    # Center word 2 (middle) should see context words 0, 1, 3, 4
    pairs_for_center_2 = [(c, ctx) for c, ctx in pairs if c == 2]
    contexts = {ctx for _, ctx in pairs_for_center_2}
    assert contexts == {0, 1, 3, 4}
