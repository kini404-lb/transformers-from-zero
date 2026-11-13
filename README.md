# transformers-from-zero

I'm a DevOps engineer learning the math behind LLMs, in public, one formula at a time.

This repo is my notebook. Every concept gets three things:

1. **Notes** in plain English, with the formula and a tiny example.
2. **NumPy code** that does exactly what the formula says, nothing hidden.
3. **Tests** that check the code reproduces the numbers in the notes.

No PyTorch, no magic. If I can't compute it by hand on three tokens, I don't understand it yet.

## The running example

Everything uses the same sentence: **"Kini likes yoga."**

```
$ python examples/kini_likes_yoga.py

2) Softmax -> attention weights (they sum to 1)
   Kini   0.090
   likes  0.245
   yoga   0.665

4) Add them up
   O_yoga = [1.421 1.155]
```

### Now with two heads

One head can only focus on one thing. Multi-head attention runs several heads
in parallel, each asking its own question, then concatenates the answers and
mixes them with an output projection `W_O`:

$$\text{MultiHead}(X) = [\ \text{head}_1 \mid \dots \mid \text{head}_h\ ]\ W_O \qquad d_{head} = d_{model} / h$$

For "yoga", head 1 looks at yoga itself, head 2 looks at Kini:

```
$ python examples/kini_likes_yoga_multi_head.py

3) Concatenate the heads side by side
   [head 1 | head 2] = [1.421 1.155 1.575 0.335]

4) Output projection: concat @ W_O
   O_yoga = [ 2.996  1.489 -0.155  0.82 ]
```

Full walkthrough in [notes/02](notes/02-multi-head-attention.md).

## Quick start

```bash
git clone https://github.com/kini404-lb/transformers-from-zero.git
cd transformers-from-zero
pip install -r requirements.txt

python examples/kini_likes_yoga.py              # one head
python examples/kini_likes_yoga_multi_head.py   # two heads
pytest                               # check them
```

Needs Python 3.9+ and NumPy.

## Roadmap

| #  | Topic                                              | Notes | Code | Status      |
|----|----------------------------------------------------|-------|------|-------------|
| 01 | Single-head attention (Q, K, V, softmax, masking)  | [notes](notes/01-single-head-attention.md) | [attention.py](tfz/attention.py) | Done |
| 02 | Multi-head attention, concatenation, output projection | [notes](notes/02-multi-head-attention.md) | [multi_head.py](tfz/multi_head.py) | Done |
| 03 | Residual connections and RMSNorm                   | | | Next |
| 04 | Feed-forward network                               | | | Planned |
| 05 | A full transformer block                           | | | Planned |

## Layout

```
notes/      plain-English notes, one file per concept
tfz/        the NumPy code ("transformers from zero")
examples/   runnable scripts that print the worked numbers
tests/      pytest checks for the code and the notes' numbers
```

## Follow along

I post each step as I learn it. If something here is wrong or unclear,
open an issue. Explaining it better is half the point.

## License

MIT
