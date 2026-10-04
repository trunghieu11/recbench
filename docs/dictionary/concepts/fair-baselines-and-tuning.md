# Fair baselines and tuning

## Why it matters

Papers often show a new model beating baselines. Replication studies found that many such wins vanish when
the baselines are tuned with the same care as the new model. A benchmark is only fair if every method gets a
comparable chance.

## Intuition

Comparing a race car tuned for months with a car straight from the factory says nothing about the designs.
The same holds for a deep model with tuned hyperparameters against an untuned EASE.

## What the research found

- Ferrari Dacrema, Cremonesi and Jannach (2019) tried to reproduce 18 neural recommendation methods. Only a
  few could be reproduced, and most of those lost to simple, well-tuned nearest-neighbour or linear baselines.
- Rendle, Zhang and Koren (2019) showed that classic baselines (for example matrix factorisation on MovieLens)
  had been reported far below their real performance, making newer methods look better than they were.

## How recbench handles it

| Aspect | Smoke and full tiers (default settings) | Quick-tier bake-off (tuned) |
|---|---|---|
| Same data, split, users and metrics | yes, for every method | yes, for every method |
| Simple baselines on the leaderboard | yes (rungs 0–2) | yes |
| Hyperparameter tuning | **no**: fixed defaults per preset | **yes**: 10 settings per method (Optuna), scored on a validation fold, the same budget for every method |
| Training length | a fixed number of steps for the older models; epochs with a cap for the newer ones | epochs with early stopping on the validation fold, within each method's limit from its paper |
| Random seeds | one (bootstrap intervals cover user sampling only) | one per trial; each dataset's top 3 are confirmed on full data with 3 seeds when training is random |
| Time limit | per method (`timeout_minutes`) | 3 hours per job, tuning included |

What this means when you read results:

- Methods with **few hyperparameters** (MostPopular, ItemKNN, EASE, iALS) lose little without tuning; deep models
  lose a lot. That is why the untuned [leaderboards](../../results/leaderboards.md) understate neural models, and why
  the [quick-tier bake-off](../../results/quick-tier.md) exists.
- On the small laptop presets, deep models are **under-trained**. The XSimGCL page shows how a longer budget and one
  changed setting took it from zero to LightGCN's level.
- Treat a deep model losing an untuned comparison as "not shown to help *with these settings*", not as "useless".

## A small example: why the budget matters

Training curves cross. With 400 steps, model A (few parameters) may already be near its best while model B
(many parameters) is still improving. With 10,000 steps their order can flip. A fair comparison therefore
reports the budget, which recbench logs with every run: the preset, `max_steps` or the epochs run
(`fit.epochs_run`, `fit.best_epoch`, and why training stopped, `fit.stopped`), and the training time.

## Pitfalls

- **Tuning only your favourite model.**
- **Tuning on the test set:** every look at test results while choosing settings leaks information. Use the
  validation window.
- **Comparing against numbers copied from papers** that used a different split or protocol.

## Check your understanding

??? question "Why does EASE lose less from 'no tuning' than SASRec?"
    It has one hyperparameter (λ), whose default works across many datasets, and a closed-form fit. SASRec
    has many interacting knobs and depends on the training length.

??? question "Where should hyperparameters be tuned in recbench's split?"
    On the validation window (between `valid_start` and `test_start`), never on the test window.

## Further reading

- Ferrari Dacrema et al. (2019), [Are We Really Making Much Progress?](https://arxiv.org/abs/1907.06902) (RecSys 2019).
- Rendle, Zhang and Koren (2019), [On the Difficulty of Evaluating Baselines](https://arxiv.org/abs/1905.01395).
- [The method ladder](../algorithms/index.md).
