"""Reproduce the "Kini likes yoga" numbers from the notes.

Run it from the repo root:

    python examples/kini_likes_yoga.py

We follow one token, "yoga", through the last two steps of attention:
softmax, then the weighted sum of Value vectors.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tfz import causal_mask, scaled_dot_product_attention, softmax  # noqa: E402

TOKENS = ["Kini", "likes", "yoga"]

# Scaled scores of "yoga" (its Query) against every token's Key.
YOGA_SCORES = np.array([1.0, 2.0, 3.0])

# Tiny 2-dimensional Value vectors, one per token.
V = np.array(
    [
        [1.0, 0.0],  # V_Kini
        [0.0, 2.0],  # V_likes
        [2.0, 1.0],  # V_yoga
    ]
)


def main():
    np.set_printoptions(precision=3, suppress=True)

    print('Sentence: "Kini likes yoga."  Following the token: yoga\n')

    print("1) Scaled scores (yoga's Query vs. each Key)")
    for tok, s in zip(TOKENS, YOGA_SCORES):
        print(f"   {tok:<6} {s:.0f}")

    weights = softmax(YOGA_SCORES)
    print("\n2) Softmax -> attention weights (they sum to 1)")
    for tok, w in zip(TOKENS, weights):
        print(f"   {tok:<6} {w:.3f}")
    print(f"   sum    {weights.sum():.3f}")

    print("\n3) Multiply each Value vector by its weight")
    parts = weights[:, None] * V
    for tok, w, v, p in zip(TOKENS, weights, V, parts):
        print(f"   {w:.3f} x V_{tok:<6} {v} = {p}")

    out = parts.sum(axis=0)
    print("\n4) Add them up")
    print(f"   O_yoga = {out}   (the notes round this to [1.42, 1.155])")

    # Same thing for the whole sentence at once, with a causal mask.
    # We build a full 3x3 score matrix whose last row is yoga's scores.
    print("\n5) Bonus: the whole sentence at once, with a causal mask")
    S = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            YOGA_SCORES,
        ]
    )
    A = softmax(S + causal_mask(3))
    print("   Attention weights A (row = token doing the looking):")
    for tok, row in zip(TOKENS, A):
        print(f"   {tok:<6} {row}")
    print("   Kini can only see itself, likes sees Kini and likes, yoga sees all three.")
    print(f"   Output for yoga: {(A @ V)[2]}")

    # And a fully random version, starting from embeddings, to show the shapes.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 4))  # 3 tokens, embedding size 4
    W_Q, W_K, W_V = (rng.normal(size=(4, 2)) for _ in range(3))
    O, A = scaled_dot_product_attention(X @ W_Q, X @ W_K, X @ W_V, causal=True)
    print("\n6) From random embeddings: X (3x4) -> Q, K, V (3x2) -> O (3x2)")
    print(f"   O shape = {O.shape}, every row of A sums to {A.sum(axis=1)}")


if __name__ == "__main__":
    main()
