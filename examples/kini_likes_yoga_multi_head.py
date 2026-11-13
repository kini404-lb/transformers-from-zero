"""Reproduce the multi-head "Kini likes yoga" numbers from notes/02.

Run it from the repo root:

    python examples/kini_likes_yoga_multi_head.py

We follow the token "yoga" through two attention heads, glue the two
results together, then mix them with the output projection W_O.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tfz import (  # noqa: E402
    merge_heads,
    multi_head_attention,
    multi_head_attention_loop,
    softmax,
    split_heads,
)

TOKENS = ["Kini", "likes", "yoga"]

# Head 1 is the head from notes/01: same scores, same Value vectors.
HEAD1_SCORES = np.array([1.0, 2.0, 3.0])
HEAD1_V = np.array(
    [
        [1.0, 0.0],  # V1_Kini
        [0.0, 2.0],  # V1_likes
        [2.0, 1.0],  # V1_yoga
    ]
)

# Head 2 has its own W_Q, W_K, W_V, so it scores the tokens differently.
# Here it looks mostly at "Kini" (who is doing the yoga?).
HEAD2_SCORES = np.array([3.0, 1.0, 2.0])
HEAD2_V = np.array(
    [
        [2.0, 0.0],  # V2_Kini
        [0.0, 1.0],  # V2_likes
        [1.0, 1.0],  # V2_yoga
    ]
)

# Output projection, shape (4, 4). Columns 1-2 add the heads, 3-4 subtract them.
W_O = np.array(
    [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, -1.0],
    ]
)


def show_head(name, scores, V):
    weights = softmax(scores)
    out = weights @ V
    print(f"\n{name}")
    for tok, s, w, v in zip(TOKENS, scores, weights, V):
        print(f"   {tok:<6} score {s:.0f}  weight {w:.3f}  x {v}")
    print(f"   output = {out}")
    return out


def main():
    np.set_printoptions(precision=3, suppress=True)

    print('Sentence: "Kini likes yoga."  Following the token: yoga')

    o1 = show_head("1) Head 1 (the head from notes/01, likes yoga itself)", HEAD1_SCORES, HEAD1_V)
    o2 = show_head("2) Head 2 (a second head, cares about Kini)", HEAD2_SCORES, HEAD2_V)

    concat = np.concatenate([o1, o2])
    print("\n3) Concatenate the heads side by side")
    print(f"   [head 1 | head 2] = {concat}")

    out = concat @ W_O
    print("\n4) Output projection: concat @ W_O")
    print(f"   O_yoga = {out}")
    print("   (first two numbers: head 1 + head 2, last two: head 1 - head 2)")

    # The shapes, from random embeddings, with the real layer.
    rng = np.random.default_rng(0)
    n, d_model, h = 3, 8, 2
    X = rng.normal(size=(n, d_model))
    W_Q, W_K, W_V, W_O_rand = (rng.normal(size=(d_model, d_model)) for _ in range(4))

    print(f"\n5) The full layer on random embeddings: n={n}, d_model={d_model}, h={h}")
    Q = X @ W_Q
    print(f"   X           {X.shape}")
    print(f"   Q = X W_Q   {Q.shape}")
    print(f"   split heads {split_heads(Q, h).shape}   (h, n, d_head)")
    O, A = multi_head_attention(X, W_Q, W_K, W_V, W_O_rand, num_heads=h, causal=True)
    print(f"   weights A   {A.shape}   one (n, n) table per head")
    print(f"   merge heads {merge_heads(split_heads(Q, h)).shape}")
    print(f"   output O    {O.shape}   same shape as X")

    O_loop, _ = multi_head_attention_loop(X, W_Q, W_K, W_V, W_O_rand, num_heads=h, causal=True)
    print(f"   vectorised == for-loop version: {np.allclose(O, O_loop)}")


if __name__ == "__main__":
    main()
