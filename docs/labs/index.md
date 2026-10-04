# Improve the light methods yourself

!!! abstract "In plain words"
    Eleven weeks, eleven methods, one at a time, from the simplest to the most involved. For each method you
    reproduce its result, learn what every setting does, try ways to feed it better data, and implement one idea from
    the research literature. Then you judge the idea with a fair, paired test, and if it wins, make it part of
    recbench. All eleven are light: they train in seconds to minutes on this laptop, need no GPU, and need no
    PyTorch.

## The plan

| Week | Method | What it teaches | Primer to read first |
|---|---|---|---|
| 0 | the tools | set up, the baseline, `once`, `sweep`, `compare` | [sparse matrices](primer-sparse-matrices.md), [comparing methods fairly](primer-comparing.md) |
| 1 | [ItemKNN](01-itemknn.md) | the user × item matrix, item similarity, recency | |
| 2 | [RP3beta](02-rp3beta.md) | recommendation as a random walk; penalising popularity | |
| 3 | [EASE](03-ease.md) | a model learned in one formula: one matrix inverse | [linear algebra, part 1](primer-linear-algebra.md) |
| 4 | [SLIM](04-slim.md) | the same idea as many small sparse regressions; L1 and L2 penalties | [optimisation](primer-optimisation.md) |
| 5 | [SANSA](05-sansa.md) | approximating a big inverse cheaply; accuracy against time | |
| 6 | [PureSVD](06-puresvd.md) | low-rank "taste directions" (SVD) | [linear algebra, part 2](primer-linear-algebra.md#part-2-svd-and-filters) |
| 7 | [GF-CF](07-gfcf.md) | graph filters: smoothing over the user-item graph, without training | |
| 8 | [iALS](08-ials.md) | the first iterative model: alternating least squares, confidence weights | |
| 9 | [BPR-MF](09-bpr-mf.md) | stochastic gradient descent and a pairwise loss; writing your own training loop | [optimisation: SGD](primer-optimisation.md#stochastic-gradient-descent) |
| 10 | [V-SKNN](10-vsknn.md) | order and sessions: the user's last few clicks | |
| 11 | [LightGBM re-ranker](11-lgbm-rerank.md) | two stages: combining methods, then learning to rank | [trees and learning to rank](primer-trees-and-ranking.md) |

**Why this order.** Each method reuses an idea from the one before. ItemKNN counts co-occurrences; RP3beta turns the
same counts into a random walk; EASE learns the item-to-item weights in one formula; SLIM learns them as regressions;
SANSA approximates EASE for big catalogs. PureSVD and GF-CF move from items to "directions" in the data. iALS and
BPR-MF learn user and item vectors by iterating. V-SKNN adds the order of clicks. The LightGBM re-ranker comes last
because it combines EASE and ItemKNN: improving them first also improves it.

**Time.** About 5 to 8 hours per week of reading, coding and analysis, plus the experiment runs, which go to a
rented box (MovieLens jobs take minutes; the big datasets up to a few hours each).

## Each lab: four levels

| Level | You | About |
|---|---|---|
| 1. Reproduce and read | reproduce the baseline's best validation score exactly; read the method's code with guiding questions; look inside the fitted model | 1 hour |
| 2. Understand each setting | predict how each setting changes the results, sweep it on the validation fold, explain what you see | 1.5 hours |
| 3. Data tricks | change *what* the method learns from: recency, counts instead of 0/1, rare items, sessions | 1.5 hours |
| 4. One change from the literature | implement an idea from a paper as a variant, test it, run it on all five datasets, compare, and analyse where it helped | 2 to 3 hours |

Each level has exercises with **hints** and **solutions**, folded so that you open them only when needed. Try
first: you learn most from the attempt, even a failed one. Every lab has a notebook,
`labs/<nn>-<method>/<method>.ipynb`, with the code cells for its exercises, and an `experiments.yaml` for its
experiments. The solutions were checked on a separate copy of the code. The `main` branch still holds the baseline,
so the improvements are yours to make.

## How an improvement is judged

1. **The baseline** is the bake-off's own tuning, repeated for every lab method and dataset: 10 settings tried on the
   validation fold, the best one tested once on the test split. Random methods are tested with 3 seeds and averaged.
   It runs on a rented box ([set up, step 5](../handbook/setup.md#step-5-the-baseline)); the
   [scoreboard](scoreboard.md) shows it.
2. **An experiment** gets the same budget and the same procedure (see [run experiments](../handbook/run-experiments.md)).
3. **Better on a dataset** means the paired 95% interval of B − A in test NDCG@10 is above zero
   ([compare results](../handbook/compare-results.md)). Accuracy decides; training and scoring time are reported next
   to it.
4. **Promotion:** better on all five datasets makes it the new default; better on some and worse on none adds it as a
   search-space option; worse anywhere means no promotion ([promote a winner](../handbook/promote.md)).

## Week 0: before lab 1

- [ ] [Set up](../handbook/setup.md): the lab extra, VS Code, the notebooks, `./scripts/check.sh --quick`.
- [ ] Compute the baseline on a rented box and fetch it ([set up, step 5](../handbook/setup.md#step-5-the-baseline)).
- [ ] Read the primers on [sparse matrices](primer-sparse-matrices.md) and on [comparing methods
      fairly](primer-comparing.md).
- [ ] Open the [scoreboard](scoreboard.md). Where does each method start? Which methods are below MostPopular, the
      floor, on which datasets?
- [ ] Practise the tools on MovieLens (these fit small models; on the box, or on the laptop if you allow it there):
      `python -m recbench.lab once --method itemknn --dataset movielens-25m --best`,
      `python -m recbench.lab sweep --method itemknn --dataset movielens-25m --param knn_shrink --values 0,10,100,1000`,
      and `python -m recbench.compare most_popular itemknn --segments`.
- [ ] Read the [Git workflow](../handbook/git-workflow.md) and create your first branch, `lab/01-itemknn`.

## Your progress

The [scoreboard](scoreboard.md) shows every method's baseline and your best experiment so far on each dataset. It is
rebuilt by `python -m recbench.lab scoreboard --docs`, and each lab page starts with its method's part of it.
