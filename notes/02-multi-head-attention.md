# 02: Multi-head attention

In [01](01-single-head-attention.md) one attention head turned "yoga" into one new vector.
One head gives every token **one** set of attention weights, so it can only
focus on one kind of relationship at a time.

Multi-head attention runs **several heads side by side**, each with its own
`W_Q`, `W_K`, `W_V`, then glues their answers together. Same sentence as before:
**"Kini likes yoga."**

Code for every step lives in [`tfz/multi_head.py`](../tfz/multi_head.py).
The worked numbers are in [`examples/kini_likes_yoga_multi_head.py`](../examples/kini_likes_yoga_multi_head.py).

---

## Part 1. Why more than one head?

Think about what "yoga" might want to know about the other tokens:

- *What is being liked?* It should look at itself and at "likes".
- *Who is doing it?* It should look at "Kini".

One head has one row of softmax weights for "yoga". If it puts 66% on itself,
there is only 34% left for everything else. With two heads, each one can spend
its full 100% on a different question.

Nobody tells a head which question to ask. Each head learns its own weights in
training, and different heads usually end up looking for different patterns.

---

## Part 2. The formulas

For each head *i*, from 1 to *h*:

$$\text{head}_i = \text{Attention}(XW_Q^{(i)},\ XW_K^{(i)},\ XW_V^{(i)})$$

This is exactly the single-head attention from 01, just with smaller weight matrices.
Then concatenate the heads and apply one more learned matrix, the **output projection**:

$$\text{MultiHead}(X) = [\ \text{head}_1 \mid \text{head}_2 \mid \dots \mid \text{head}_h\ ]\ W_O$$

### Sizes

The heads share the work instead of making the layer bigger:

$$d_{head} = \frac{d_{model}}{h}$$

| Thing                       | Shape                  |
|-----------------------------|------------------------|
| `X`                         | `(n, d_model)`         |
| one head's `W_Q`, `W_K`, `W_V` | `(d_model, d_head)` |
| one head's output           | `(n, d_head)`          |
| all heads concatenated      | `(n, h * d_head) = (n, d_model)` |
| `W_O`                       | `(d_model, d_model)`   |
| final output                | `(n, d_model)`, same shape as `X` |

Example: GPT-2 small has `d_model = 768` and `h = 12`, so every head works with
vectors of size `64`.

### The trick: one big matrix, cut into slices

Stack every head's `W_Q` side by side and you get one `(d_model, d_model)` matrix.
So in code we do **one** big multiplication, `Q = X @ W_Q`, then cut the columns
of `Q` into `h` slices. Slice 1 is head 1's Queries, slice 2 is head 2's, and so on.

```
Q = X W_Q          (n, d_model)
    [ q q | q q ]  <- columns 0-1 go to head 1, columns 2-3 go to head 2
```

That is all `split_heads` does: a reshape, no new numbers. `merge_heads` puts the
slices back side by side. The same total number of parameters as one big head,
just used as `h` smaller, independent heads.

---

## Part 3. Worked example: "yoga" with two heads

Let `d_model = 4` and `h = 2`, so each head has `d_head = 2`.

### Head 1 (the head from 01)

Same scaled scores and Value vectors as notes 01:

| Token | Scaled score | Weight | Value     |
|-------|-------------:|-------:|-----------|
| Kini  | 1            | 0.090  | `[1, 0]`  |
| likes | 2            | 0.245  | `[0, 2]`  |
| yoga  | 3            | 0.665  | `[2, 1]`  |

$$\text{head}_1 = [1.42,\ 1.155]$$

### Head 2 (a new head with its own weights)

Head 2 has different `W_Q`, `W_K`, `W_V`, so it scores the tokens differently.
This one looks mostly at "Kini":

| Token | Scaled score | Weight | Value     |
|-------|-------------:|-------:|-----------|
| Kini  | 3            | 0.665  | `[2, 0]`  |
| likes | 1            | 0.090  | `[0, 1]`  |
| yoga  | 2            | 0.245  | `[1, 1]`  |

```
0.665 x [2, 0] = [1.330, 0.000]
0.090 x [0, 1] = [0.000, 0.090]
0.245 x [1, 1] = [0.245, 0.245]
```

$$\text{head}_2 = [1.575,\ 0.335]$$

### Concatenate

Put the two head outputs side by side:

$$[\ \text{head}_1 \mid \text{head}_2\ ] = [1.42,\ 1.155,\ 1.575,\ 0.335]$$

Back to size 4, the same size as the embedding.

### Output projection

Concatenating just stacks the heads next to each other. They still don't talk.
`W_O` mixes them, so every output number can use what every head found.
A tiny `W_O`, chosen to be easy to follow:

```
        [ 1  0   1   0 ]
W_O  =  [ 0  1   0   1 ]
        [ 1  0  -1   0 ]
        [ 0  1   0  -1 ]
```

Columns 1 and 2 add head 1 and head 2. Columns 3 and 4 subtract them:

```
[1.42 + 1.575,  1.155 + 0.335,  1.42 - 1.575,  1.155 - 0.335]
```

$$O_{yoga} = [2.996,\ 1.49,\ -0.155,\ 0.82]$$

That is the multi-head representation of "yoga". In a real model `W_O` is
learned, not hand-picked like this.

Run it yourself:

```bash
python examples/kini_likes_yoga_multi_head.py
```

---

## Part 4. Masking

The causal mask from 01 works exactly the same: every head gets the same mask,
so no head can peek at future tokens.

---

## Where I am

Done:

- [x] Single-head attention (notes 01)
- [x] Several heads in parallel
- [x] Splitting `Q`, `K`, `V` into heads
- [x] Concatenation
- [x] Output projection `W_O`

Next: **residual connections and RMSNorm**, the glue that lets us stack
many attention layers on top of each other.

One caveat: many recent models use variants like grouped-query attention,
where several Query heads share one Key/Value head to save memory. The idea
is the same; I'll come back to it once the basic block works.
