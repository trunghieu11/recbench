# Primer: learning by optimisation

!!! abstract "In plain words"
    Many methods *learn* their weights: they define a number that measures how wrong they are (the **loss**) and
    change the weights until the loss is small. You need four ideas for the labs: **penalties** that keep weights
    small or exactly zero (SLIM, lab 4), **gradient descent** (walk downhill on the loss), **alternating least
    squares** (solve one half exactly while the other is fixed: iALS, lab 8), and **stochastic gradient descent** on
    a pairwise loss (BPR-MF, lab 9, where you write your own training loop).

## Loss and penalties

A loss compares predictions with what happened. The squared error $\sum (y - \hat y)^2$ is the classic one. A
**penalty** is added to it to prefer simpler weights:

| Penalty | Added to the loss | Effect |
|---|---|---|
| L2 (ridge) | $\lambda \sum w^2$ | all weights shrink a little; none becomes exactly 0 |
| L1 (lasso) | $\lambda \sum \lvert w \rvert$ | many weights become **exactly 0**: a sparse model |
| ElasticNet | a mix of both, set by `l1_ratio` | sparse like L1, more stable like L2 |

SLIM (lab 4) fits one ElasticNet regression per item: its weights to the other items. L1 makes most weights 0, so
each item keeps a short list of predictors and the model stays small. `slim_alpha` sets the total penalty and
`slim_l1_ratio` the mix.

## Gradient descent: walk downhill

The **gradient** of the loss says in which direction it grows fastest. Step the other way, by a **learning rate**
times the gradient, and repeat. One weight, loss $f(w) = (w - 3)^2$, gradient $2(w - 3)$, learning rate 0.25,
starting at 0:

| Step | w | gradient | new w |
|---|---|---|---|
| 1 | 0 | −6 | 1.5 |
| 2 | 1.5 | −3 | 2.25 |
| 3 | 2.25 | −1.5 | 2.625 |
| 4 | 2.625 | −0.75 | 2.8125 |

It closes in on 3, the minimum. A learning rate that is too small is slow; one that is too large jumps over the
minimum and can diverge.

## Alternating least squares (iALS)

Matrix factorisation learns a vector $p_u$ per user and $q_i$ per item so that $p_u \cdot q_i$ is high for items the
user interacted with. Learning both at once is hard. But **with the item vectors fixed**, each user vector is a ridge
regression with an exact solution (the formula from the [linear algebra primer](primer-linear-algebra.md#ridge-regression-the-best-weights-kept-small)).
Then fix the users and solve every item the same way. Alternate. Each half-step can only lower the loss.

iALS also weighs every user-item cell by a **confidence**: $c_{ui} = 1 + \alpha \cdot r_{ui}$ for an interaction
count $r_{ui}$, and 1 for every empty cell ("probably not interested, but not sure"). A tiny example: two items with
vectors $q_1 = (1, 0)$ and $q_2 = (0, 1)$, a user who interacted with item 1 once (α = 10, so confidence 11) and not
with item 2 (confidence 1), and λ = 0.1. Solving gives $p_u = (0.991, 0)$: the user points at item 1. `ials_alpha`
sets how much more an interaction counts than an empty cell; lab 8 tries a log-scaled confidence instead.

## Stochastic gradient descent

With millions of interactions, computing the loss's full gradient for every step is slow. **Stochastic** gradient
descent (SGD) takes one small random piece of data at a time, an example or a mini-batch, and makes a small step for
it. Many cheap, noisy steps get there faster than few exact ones. One pass over all the data is an **epoch** (in
`implicit`'s BPR: an *iteration*).

### BPR: a pairwise loss

BPR-MF (lab 9) does not try to predict 1s and 0s. It learns to **rank**: for a user u, an item i they interacted with
should score above an item j they did not. For one triple (u, i, j):

$$x_{uij} = p_u \cdot (q_i - q_j), \qquad \text{loss} = -\ln \sigma(x_{uij}), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$

The loss is small when $x_{uij}$ is large (i ranked well above j). The SGD update, with learning rate η and the
factor $g = \sigma(-x_{uij})$:

$$p_u \leftarrow p_u + \eta\, g\, (q_i - q_j), \qquad q_i \leftarrow q_i + \eta\, g\, p_u, \qquad q_j \leftarrow q_j - \eta\, g\, p_u$$

(each also minus η times a regularisation term, `bpr_reg`, left out here). One update by hand, with η = 0.1:

| | before | after |
|---|---|---|
| $p_u$ | (0.1, 0.2) | (0.1051, 0.1846) |
| $q_i$ (seen) | (0.3, 0.1) | (0.3051, 0.1102) |
| $q_j$ (not seen) | (0.2, 0.4) | (0.1949, 0.3898) |
| $x_{uij}$ | −0.0500 | −0.0400 |
| loss | 0.7185 | 0.7134 |

The pair moved in the right direction: i rose and j fell for this user. Training repeats this for millions of random
triples. Which j to draw, the **negative sampling**, matters a lot: a random item is usually an easy negative. Lab 9
draws popular items more often, which makes harder negatives.

## Exercises

**1.** In the gradient descent table, what happens with a learning rate of 1.0?

??? success "Solution"
    From 0: gradient −6, so w = 0 − 1.0 × (−6) = 6. Then gradient +6, w = 0, then 6 again: it jumps between 0 and 6
    forever. Above 1.0 it diverges. The learning rate must suit the loss's curvature.

**2.** Why does an L1 penalty give exact zeros and L2 does not?

??? tip "Hint"
    Compare how strongly each penalty pushes a weight that is already small, say 0.01.

??? success "Solution"
    L1's push towards 0 has the same size (λ) however small the weight is, so a small weight is pushed all the way to
    exactly 0. L2's push is proportional to the weight itself (2λw), so it fades as the weight shrinks and never
    quite reaches 0.

**3.** In the BPR update, why is the step large when $x_{uij}$ is very negative and almost zero when it is very
positive?

??? success "Solution"
    The factor $g = \sigma(-x_{uij})$ is close to 1 when $x_{uij} \ll 0$ (the pair is badly ordered: learn a lot)
    and close to 0 when $x_{uij} \gg 0$ (already ordered: nothing to learn). This is also why easy negatives teach
    little: they are already ranked low.
