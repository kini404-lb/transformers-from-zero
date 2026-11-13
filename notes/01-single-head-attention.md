# 01: Single-head attention

These are my notes on one attention calculation, from the sentence to the output vector.
I follow one example the whole way: **"Kini likes yoga."**

Code for every step lives in [`tfz/attention.py`](../tfz/attention.py).
The worked numbers are in [`examples/kini_likes_yoga.py`](../examples/kini_likes_yoga.py).

---

## Part 1. Input preparation

### 1. Tokenization

The sentence is split into tokens:

```
"Kini likes yoga."  ->  [Kini] [likes] [yoga]
```

### 2. Token embeddings

Each token becomes a vector of numbers. Stack them as rows of a matrix `X`:

```
X = [ x_Kini  ]
    [ x_likes ]
    [ x_yoga  ]
```

`X` has shape `(n, d)`: `n` tokens, each a vector of size `d`.

---

## Part 2. Query, Key and Value

We transform each embedding three different ways, using three learned weight matrices:

$$Q = XW_Q \qquad K = XW_K \qquad V = XW_V$$

| Vector | Name  | The question it answers                                  |
|--------|-------|----------------------------------------------------------|
| Q      | Query | What information am I looking for?                       |
| K      | Key   | What do I offer for other tokens to match against?       |
| V      | Value | What information can I contribute to another token?      |

Q, K and V are three different learned views of the same tokens.

---

## Part 3. Calculating attention

### 3. Attention scores (dot product)

$$S = QK^T$$

`S[i, j]` is the dot product of token *i*'s Query with token *j*'s Key.
A big number means "token *j* looks relevant to token *i*".

### 4. Scaling

$$S_{scaled} = \frac{QK^T}{\sqrt{d_k}}$$

`d_k` is the size of the Key vectors. Dot products of long vectors get large,
and very large scores make softmax put almost all the weight on one token.
Dividing by `sqrt(d_k)` keeps the scores in a sensible range.

### 5. Masking (causal attention)

$$S_{masked} = S_{scaled} + M \qquad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

In a model that writes text left to right, a token must not peek at future tokens.
Adding `-inf` to those positions means they get exactly zero weight after softmax.

```
          Kini  likes  yoga
Kini   [   0    -inf   -inf ]
likes  [   0     0     -inf ]
yoga   [   0     0      0   ]
```

### 6. Softmax

$$A_{ij} = \frac{e^{S_{ij}}}{\sum_{k} e^{S_{ik}}}$$

- Numerator: the exponential of one score.
- Denominator: the sum of the exponentials of all allowed scores in that row.
- Result: one weight per token, every row adds up to 1 (100%).

---

## Part 4. After softmax: weighted values

$$O = AV \qquad\text{or, for one token:}\qquad O_i = \sum_j A_{ij} V_j$$

Multiply each Value vector by its attention weight, then add them all up.
The result is a new, context-aware representation of the token.

### Worked example: the token "yoga"

Scaled scores of yoga's Query against each Key:

| Token | Scaled score | After softmax |
|-------|-------------:|--------------:|
| Kini  | 1            | 0.090         |
| likes | 2            | 0.245         |
| yoga  | 3            | 0.665         |

Some tiny 2-dimensional Value vectors:

```
V_Kini  = [1, 0]
V_likes = [0, 2]
V_yoga  = [2, 1]
```

Multiply each by its weight:

```
0.090 x [1, 0] = [0.090, 0.000]
0.245 x [0, 2] = [0.000, 0.490]
0.665 x [2, 1] = [1.330, 0.665]
```

Add them:

$$O_{yoga} = [1.42,\ 1.155]$$

That is the new representation of "yoga" for this attention head.
We repeat the same thing for every token, each with its own Query and weights.

Run it yourself:

```bash
python examples/kini_likes_yoga.py
```

---

## Where I am

Done:

- [x] Input embeddings
- [x] Q, K, V projections
- [x] Attention scores, scaling, masking
- [x] Softmax
- [x] Weighted values

Next: **multi-head attention** (several heads, concatenation, output projection).
After that: residual connection, RMSNorm, feed-forward network.

One caveat: this is the simplified standard transformer. Real models
change details like where normalization goes or which FFN variant they use.
