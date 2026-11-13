import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tfz import (  # noqa: E402
    merge_heads,
    multi_head_attention,
    multi_head_attention_loop,
    single_head_attention,
    softmax,
    split_heads,
)

HEAD1_V = np.array([[1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])
HEAD2_V = np.array([[2.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
W_O_EXAMPLE = np.array(
    [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, -1.0],
    ]
)


def random_layer(seed, n=3, d_model=8):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d_model))
    W = [rng.normal(size=(d_model, d_model)) for _ in range(4)]
    return X, W


# --- The worked numbers from notes/02 -------------------------------------


def test_kini_likes_yoga_head_outputs():
    o1 = softmax(np.array([1.0, 2.0, 3.0])) @ HEAD1_V
    o2 = softmax(np.array([3.0, 1.0, 2.0])) @ HEAD2_V
    np.testing.assert_allclose(o1, [1.420, 1.155], atol=1e-3)
    np.testing.assert_allclose(o2, [1.575, 0.335], atol=1e-3)


def test_kini_likes_yoga_output_projection():
    o1 = softmax(np.array([1.0, 2.0, 3.0])) @ HEAD1_V
    o2 = softmax(np.array([3.0, 1.0, 2.0])) @ HEAD2_V
    out = np.concatenate([o1, o2]) @ W_O_EXAMPLE
    np.testing.assert_allclose(out, [2.996, 1.490, -0.155, 0.820], atol=1e-3)


# --- Splitting and merging ------------------------------------------------


def test_split_heads_takes_column_slices():
    M = np.arange(12.0).reshape(3, 4)  # 3 tokens, d_model = 4
    H = split_heads(M, num_heads=2)
    assert H.shape == (2, 3, 2)
    np.testing.assert_array_equal(H[0], M[:, :2])
    np.testing.assert_array_equal(H[1], M[:, 2:])


def test_merge_heads_undoes_split_heads():
    M = np.random.default_rng(4).normal(size=(5, 12))
    for h in (1, 2, 3, 4, 6, 12):
        np.testing.assert_array_equal(merge_heads(split_heads(M, h)), M)


def test_split_heads_rejects_uneven_split():
    with pytest.raises(ValueError):
        split_heads(np.zeros((3, 10)), num_heads=3)


# --- The full layer -------------------------------------------------------


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
def test_vectorised_matches_loop(num_heads, causal):
    X, (W_Q, W_K, W_V, W_O) = random_layer(seed=num_heads)
    O, A = multi_head_attention(X, W_Q, W_K, W_V, W_O, num_heads, causal=causal)
    O_loop, A_loop = multi_head_attention_loop(
        X, W_Q, W_K, W_V, W_O, num_heads, causal=causal
    )
    np.testing.assert_allclose(O, O_loop)
    np.testing.assert_allclose(A, A_loop)


def test_one_head_with_identity_projection_is_single_head_attention():
    X, (W_Q, W_K, W_V, _) = random_layer(seed=5)
    O_multi, A_multi = multi_head_attention(
        X, W_Q, W_K, W_V, np.eye(8), num_heads=1
    )
    O_single, A_single = single_head_attention(X, W_Q, W_K, W_V)
    np.testing.assert_allclose(O_multi, O_single)
    np.testing.assert_allclose(A_multi[0], A_single)


def test_each_head_matches_single_head_on_its_slice():
    X, (W_Q, W_K, W_V, W_O) = random_layer(seed=6)
    _, A = multi_head_attention(X, W_Q, W_K, W_V, W_O, num_heads=2)
    for i, cols in enumerate([slice(0, 4), slice(4, 8)]):
        _, A_i = single_head_attention(X, W_Q[:, cols], W_K[:, cols], W_V[:, cols])
        np.testing.assert_allclose(A[i], A_i)


def test_heads_can_attend_differently():
    X, (W_Q, W_K, W_V, W_O) = random_layer(seed=7)
    _, A = multi_head_attention(X, W_Q, W_K, W_V, W_O, num_heads=2)
    assert not np.allclose(A[0], A[1])


def test_shapes_and_weights():
    X, (W_Q, W_K, W_V, W_O) = random_layer(seed=8, n=5)
    O, A = multi_head_attention(X, W_Q, W_K, W_V, W_O, num_heads=4)
    assert O.shape == (5, 8)  # same shape as X
    assert A.shape == (4, 5, 5)  # one (n, n) table per head
    np.testing.assert_allclose(A.sum(axis=-1), np.ones((4, 5)))


def test_causal_mask_applies_to_every_head():
    X, (W_Q, W_K, W_V, W_O) = random_layer(seed=9, n=4)
    _, A = multi_head_attention(X, W_Q, W_K, W_V, W_O, num_heads=4, causal=True)
    future = np.triu_indices(4, k=1)
    for head in A:
        assert np.all(head[future] == 0.0)
