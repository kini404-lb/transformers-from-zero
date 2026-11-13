import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tfz import (  # noqa: E402
    attention_scores,
    causal_mask,
    project_qkv,
    scale_scores,
    scaled_dot_product_attention,
    single_head_attention,
    softmax,
    weighted_values,
)

V_EXAMPLE = np.array([[1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])


def test_kini_likes_yoga_softmax_weights():
    weights = softmax(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(weights, [0.090, 0.245, 0.665], atol=1e-3)


def test_kini_likes_yoga_output_vector():
    weights = softmax(np.array([1.0, 2.0, 3.0]))
    out = weighted_values(weights, V_EXAMPLE)
    np.testing.assert_allclose(out, [1.42, 1.155], atol=1e-3)


def test_softmax_rows_sum_to_one():
    S = np.random.default_rng(1).normal(size=(5, 5))
    np.testing.assert_allclose(softmax(S).sum(axis=-1), np.ones(5))


def test_softmax_is_stable_for_large_scores():
    weights = softmax(np.array([1000.0, 1001.0, 1002.0]))
    assert np.all(np.isfinite(weights))
    np.testing.assert_allclose(weights, softmax(np.array([0.0, 1.0, 2.0])))


def test_causal_mask_shape_and_values():
    M = causal_mask(3)
    expected = np.array(
        [
            [0.0, -np.inf, -np.inf],
            [0.0, 0.0, -np.inf],
            [0.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_array_equal(M, expected)


def test_causal_attention_gives_zero_weight_to_future_tokens():
    rng = np.random.default_rng(2)
    Q, K, V = (rng.normal(size=(4, 3)) for _ in range(3))
    _, A = scaled_dot_product_attention(Q, K, V, causal=True)
    assert np.all(A[np.triu_indices(4, k=1)] == 0.0)
    np.testing.assert_allclose(A.sum(axis=-1), np.ones(4))
    # The first token can only look at itself.
    np.testing.assert_allclose(A[0], [1.0, 0.0, 0.0, 0.0])


def test_scaling_divides_by_sqrt_dk():
    S = np.array([[8.0, 4.0]])
    np.testing.assert_allclose(scale_scores(S, d_k=16), [[2.0, 1.0]])


def test_full_layer_matches_manual_steps():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(3, 4))
    W_Q, W_K, W_V = (rng.normal(size=(4, 2)) for _ in range(3))

    Q, K, V = project_qkv(X, W_Q, W_K, W_V)
    S = attention_scores(Q, K) / np.sqrt(2)
    expected = softmax(S) @ V

    O, A = single_head_attention(X, W_Q, W_K, W_V)
    assert O.shape == (3, 2)
    assert A.shape == (3, 3)
    np.testing.assert_allclose(O, expected)


@pytest.mark.parametrize("n", [1, 2, 5])
def test_output_shape(n):
    rng = np.random.default_rng(n)
    Q, K = rng.normal(size=(n, 4)), rng.normal(size=(n, 4))
    V = rng.normal(size=(n, 6))
    O, A = scaled_dot_product_attention(Q, K, V)
    assert O.shape == (n, 6)
    assert A.shape == (n, n)
