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

## How recbench handles it (and what it does not do yet)

| Aspect | recbench today |
|---|---|
| Same data, same split, same users, same metrics | yes, for every method |
| Simple baselines on the leaderboard | yes (rungs 0–2) |
| Hyperparameter tuning | **no**: fixed presets and step budgets, identical across methods of the same family |
| Multiple random seeds | no (bootstrap intervals cover user sampling only) |
| Training budget | a fixed number of steps per preset (400 on the laptop, 10,000 or 30,000 on the GPU) |

What this means when you read results:

- Methods with **few hyperparameters** (MostPopular, ItemKNN, EASE, iALS) lose little from the lack of tuning.
- Deep models (SASRec, HSTU, XSimGCL) usually need tuning and long training. On a 400-step laptop budget they
  are **under-trained**. The XSimGCL page shows how a longer budget and one changed hyperparameter took it
  from zero to LightGCN's level.
- Treat a deep model losing on the smoke tier as "not shown to help *at this budget*", not "useless".

Tuning on the validation window (with Optuna) and multi-seed runs are on the [roadmap](../../results/roadmap.md).

## A small example: why the budget matters

Training curves cross. With 400 steps, model A (few parameters) may already be near its best while model B
(many parameters) is still improving. With 10,000 steps their order can flip. A fair comparison therefore
reports the budget, which recbench logs with every run (`max_steps`, the preset name, and training time).

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
