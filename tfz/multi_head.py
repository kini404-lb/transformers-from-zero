"""Multi-head attention, written step by step in NumPy.

Every function here maps to one step in notes/02-multi-head-attention.md.
Shapes use these letters:

    n       = number of tokens in the sentence
    d_model = size of each token embedding (and of the layer's output)
    h       = number of heads
    d_head  = size of each head's Query / Key / Value vectors = d_model / h
"""

import numpy as np

from .attention import causal_mask, scaled_dot_product_attention, softmax


def split_heads(M, num_heads):
    """Step 2: cut the columns of M into num_heads equal slices.

    M has shape (n, d_model). Returns shape (h, n, d_head), where
    head 0 gets columns [0, d_head), head 1 gets [d_head, 2 * d_head), and so on.
    No numbers change: it is only a reshape.
    """
    n, d_model = M.shape
    if d_model % num_heads != 0:
        raise ValueError(
            f"d_model={d_model} must be divisible by num_heads={num_heads}"
        )
    d_head = d_model // num_heads
    return M.reshape(n, num_heads, d_head).transpose(1, 0, 2)


def merge_heads(H):
    """Step 4: concatenate the heads back side by side.

    H has shape (h, n, d_head). Returns shape (n, h * d_head), i.e.
    [ head_0 | head_1 | ... ] for every token. The exact inverse of split_heads.
    """
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)


def multi_head_attention(X, W_Q, W_K, W_V, W_O, num_heads, causal=False):
    """The whole multi-head attention layer, all heads at once.

    1. Q = X W_Q, K = X W_K, V = X W_V          each (n, d_model)
    2. split each into heads                    each (h, n, d_head)
    3. attention inside every head, in parallel (h, n, d_head)
    4. concatenate the heads                    (n, d_model)
    5. output projection: O = concat @ W_O      (n, d_model)

    Returns (O, A): the output vectors and the attention weights of
    every head, A with shape (h, n, n).
    """
    Q = split_heads(X @ W_Q, num_heads)
    K = split_heads(X @ W_K, num_heads)
    V = split_heads(X @ W_V, num_heads)

    d_head = Q.shape[-1]
    S = Q @ K.transpose(0, 2, 1) / np.sqrt(d_head)  # (h, n, n)
    if causal:
        S = S + causal_mask(S.shape[-1])  # same mask for every head
    A = softmax(S, axis=-1)
    heads = A @ V  # (h, n, d_head)

    return merge_heads(heads) @ W_O, A


def multi_head_attention_loop(X, W_Q, W_K, W_V, W_O, num_heads, causal=False):
    """Same result as multi_head_attention, written as a plain for loop.

    Easier to read: each head is just single-head attention on its own
    slice of the columns of W_Q, W_K and W_V. The tests check both agree.
    """
    d_model = W_Q.shape[1]
    if d_model % num_heads != 0:
        raise ValueError(
            f"d_model={d_model} must be divisible by num_heads={num_heads}"
        )
    d_head = d_model // num_heads

    outputs, weights = [], []
    for i in range(num_heads):
        cols = slice(i * d_head, (i + 1) * d_head)
        O_i, A_i = scaled_dot_product_attention(
            X @ W_Q[:, cols], X @ W_K[:, cols], X @ W_V[:, cols], causal=causal
        )
        outputs.append(O_i)
        weights.append(A_i)

    concat = np.concatenate(outputs, axis=-1)  # (n, d_model)
    return concat @ W_O, np.stack(weights)
