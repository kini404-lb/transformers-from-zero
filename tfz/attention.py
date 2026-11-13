"""Single-head scaled dot-product attention, written step by step in NumPy.

Every function here maps to one step in notes/01-single-head-attention.md.
Shapes use these letters:

    n   = number of tokens in the sentence
    d   = size of each token embedding
    d_k = size of each Query / Key vector
    d_v = size of each Value vector
"""

import numpy as np


def project_qkv(X, W_Q, W_K, W_V):
    """Step 2: turn token embeddings into Queries, Keys and Values.

    Q = X @ W_Q   "what is each token looking for?"
    K = X @ W_K   "what does each token offer for matching?"
    V = X @ W_V   "what information can each token contribute?"

    X has shape (n, d). Returns Q (n, d_k), K (n, d_k), V (n, d_v).
    """
    return X @ W_Q, X @ W_K, X @ W_V


def attention_scores(Q, K):
    """Step 3: raw relevance scores, S = Q @ K^T, shape (n, n).

    S[i, j] is how well token i's Query matches token j's Key.
    """
    return Q @ K.T


def scale_scores(S, d_k):
    """Step 4: divide by sqrt(d_k) so scores don't grow too large."""
    return S / np.sqrt(d_k)


def causal_mask(n):
    """Step 5: the causal mask M, shape (n, n).

    M[i, j] = 0     if j <= i  (token i may look at token j)
    M[i, j] = -inf  if j > i   (token j is in the future, so it is hidden)
    """
    future = np.triu(np.ones((n, n), dtype=bool), k=1)
    return np.where(future, -np.inf, 0.0)


def softmax(S, axis=-1):
    """Step 6: turn scores into weights that are positive and sum to 1.

    A[i, j] = exp(S[i, j]) / sum_k exp(S[i, k])

    We subtract the row maximum first. It does not change the result
    (it cancels out in the fraction) but it stops exp() from overflowing.
    exp(-inf) is 0, so masked positions get exactly 0 weight.
    """
    S = np.asarray(S, dtype=float)
    shifted = S - np.max(S, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def weighted_values(A, V):
    """Step 7: O = A @ V.

    For one token i: O[i] = sum_j A[i, j] * V[j].
    Each row of O is a new, context-aware version of that token.
    """
    return A @ V


def scaled_dot_product_attention(Q, K, V, causal=False):
    """Steps 3 to 7 in one call.

    Returns (O, A): the output vectors and the attention weights.
    """
    d_k = Q.shape[-1]
    S = scale_scores(attention_scores(Q, K), d_k)
    if causal:
        S = S + causal_mask(S.shape[0])
    A = softmax(S, axis=-1)
    return weighted_values(A, V), A


def single_head_attention(X, W_Q, W_K, W_V, causal=False):
    """The whole single-head attention layer, from embeddings to output."""
    Q, K, V = project_qkv(X, W_Q, W_K, W_V)
    return scaled_dot_product_attention(Q, K, V, causal=causal)
