# Random

> Gives every item a random score. It learns nothing, which is exactly why it is useful: any method that does
> not clearly beat it has learned nothing either.

!!! abstract "In plain words"
    Random gives every item a random score, so every user gets a random list. Nobody would ship it. It is the
    floor: any method that cannot beat Random clearly has learned nothing, or has a bug.

--8<-- "generated/methods/random.md"

!!! tip "When to use it"
    - As the **floor** of every leaderboard.
    - To sanity-check an evaluation pipeline: if Random scores well, the protocol is too easy (for
      example, too few candidate items) or something leaks.
    - As a reminder of the scale of a metric: Random's NDCG@10 depends on the catalog size.

!!! warning "When not to"
    - Never in production, except as a small "exploration" slice that collects unbiased data.

## 1. Intuition

Imagine a shop assistant who has never met you and has not looked at sales numbers. Asked for ten
suggestions, they pick ten products blindfolded. Sometimes one is right by luck. How often depends only on
how many products there are and how many of them you would have liked.

## 2. A tiny worked example

A catalog has 6 items. A user has already interacted with 2 of them, which the evaluator removes, so 4
candidates remain. Exactly 1 of the 4 is the item the user interacted with next (the relevant item).

Random puts the relevant item at each of the 4 positions with equal probability (1/4 each):

| K | Probability the relevant item is in the top K (HitRate@K) |
|---|---|
| 1 | 1/4 = 0.25 |
| 2 | 2/4 = 0.50 |
| 4 | 4/4 = 1.00 |

In general, with one relevant item among $N$ candidates, Random's expected HitRate@K is $K/N$. With a
catalog of 20,000 items, HitRate@10 is about 0.0005. Any real method should be orders of magnitude above that.

## 3. How it works

1. For each user, draw one uniform random number per item, using a random generator seeded with
   (run seed, user index). The same user always gets the same "random" list.
2. Sort the items by that number.

```mermaid
flowchart LR
    U[user index + seed] --> R[random generator]
    R --> S[one score per item]
    S --> T[top-K list]
```

## 4. The math, symbol by symbol

$$
s_{ui} \sim \mathcal{U}(0, 1), \qquad \mathbb{E}[\text{HitRate@}K] = \frac{K}{N} \text{ for one relevant item}
$$

| Symbol | Meaning | Shape / range |
|---|---|---|
| $s_{ui}$ | score of item $i$ for user $u$ | a number in [0, 1) |
| $\mathcal{U}(0,1)$ | the uniform distribution between 0 and 1 | — |
| $K$ | length of the recommended list | e.g. 10 |
| $N$ | number of candidate items after removing seen and padding items | e.g. 20,000 |

## 5. Training and inference

- **Training:** nothing to train; `fit` only stores the seed.
- **Inference:** one random vector of length (catalog size + 1) per user. Time and memory O(catalog size).
- **Hardware:** any CPU.

## 6. Hyperparameters

| Name in recbench config | What it does | Default | Typical range | Tip |
|---|---|---|---|---|
| `seed` | makes the random lists reproducible | 42 | any integer | keep it fixed when comparing runs |

## 7. In recbench

- Code: `src/recbench/methods/baselines.py::RandomRec`.
- `score_users` builds a generator `np.random.default_rng((seed, user))` per user, so results do not
  depend on batch order.
- Its spec says `scores_cold_items=True` (it happily "scores" brand-new items) and
  `handles_cold_users=True`, so it is also evaluated on users with no history (the `cold_users/*` metrics).
- Explanations always say "Picked at random." They are never personal, so its personal-explanation rate is 0.

!!! info "Fidelity"
    Faithful: there is nothing to simplify.

## 8. Results in this benchmark

Random is the zero line of every leaderboard. Its numbers on each dataset:

--8<-- "generated/methods/random-results.md"

## 9. Strengths and weaknesses

- **Strengths:** free, unbiased, and a perfect sanity check.
- **Weaknesses:** useless recommendations. It does have very high coverage and novelty, which is a
  reminder that [beyond-accuracy metrics](../metrics/novelty-diversity-serendipity.md) must always be read
  next to accuracy.

## 10. Common pitfalls

- Comparing Random across datasets: its scores depend on catalog size, so they are only meaningful within
  one dataset.
- Forgetting to remove already-seen items: Random would then waste slots on items the user already has.
  The evaluator does this for every method.

## 11. Check your understanding

??? question "With 1 relevant item among 500 candidates, what HitRate@10 do you expect from Random?"
    $10/500 = 0.02$. On average, 2% of users will see their item in a random top 10.

??? question "Why does Random have very high coverage?"
    Each user gets a different random list, so over thousands of users almost every item appears somewhere.
    High coverage alone therefore does not mean a method is good.

??? question "Why seed the generator per user instead of once per run?"
    So a user's list does not change when users are scored in a different order or batch size. That keeps
    results reproducible.

## 12. Further reading

- The [evaluation protocols](../concepts/evaluation-protocols.md) page explains why full-catalog ranking
  makes Random very weak, and why sampled metrics make it look much better than it is.
