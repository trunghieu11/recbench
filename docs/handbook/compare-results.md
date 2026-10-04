# Compare results

!!! abstract "In plain words"
    "B scores 0.2005 and A scores 0.1737, so B is better" is not enough: with another sample of users the order might
    flip. `python -m recbench.compare` answers properly. It takes the **same test users** for both, computes each
    user's difference B − A, and asks how much the average difference could wobble (a 95% interval). If the whole
    interval is above zero, B is **better**. `--segments` then shows *where* the difference comes from: which users
    and which items.

## The command

```bash
python -m recbench.compare most_popular rp3beta --datasets movielens-25m
```

A and B are written `method[:experiment]`; without `:experiment` it means the method's baseline. So
`itemknn itemknn:no-time-knobs` compares an experiment against its baseline, and `most_popular rp3beta` compares
two methods' baselines. Without `--datasets` it compares all five.

!!! success "You should see"
    ```text
    A = most_popular   B = rp3beta   test ndcg_at_10, paired over the same users (95% interval of B - A)

    dataset          users        A        B    B - A  95% interval          users +/-  verdict
    movielens-25m     1862   0.1737   0.2005  +0.0268  [+0.0210, +0.0326]   37%/30%    better

    dataset           validation A -> B    train s A -> B  score s/1k A -> B  coverage A -> B  seeds
    movielens-25m   0.2221 -> 0.2279     0.0 -> 3.2        0.01 -> 0.14    0.002 -> 0.006  1/1

    Promotion rule: better on 1, worse on 0, out of 1 compared. Compare on all 5 datasets before deciding.
    ```

| Column | Meaning |
|---|---|
| `users` | test users scored by both (the same people on both sides) |
| `A`, `B` | each side's test NDCG@10, averaged over those users (and over 3 seeds for random methods) |
| `B - A` and its interval | the average per-user difference, and a 95% bootstrap interval for it (2,000 resamples of the users) |
| `users +/-` | the share of users for whom B is higher / lower (the rest tie, mostly at 0) |
| `verdict` | **better**: the interval is above 0; **worse**: below 0; **no clear difference**: it contains 0 |
| `validation A -> B` | the validation scores that chose each side's setting (not comparable with test scores) |
| `train s`, `score s/1k`, `coverage` | the cost and the side effects, for the [accuracy-first, speed-reported](../labs/index.md#how-an-improvement-is-judged) rule |
| `seeds` | test runs averaged per side |

The 37% and 30% do not add up to 100%: for the other 33% of users both sides score the same, almost always 0 (neither
method found any of their test items in its top 10).

## Why "paired" matters

Six users, NDCG@10 per user:

| user | A | B | B − A |
|---|---|---|---|
| 1 | 0.00 | 0.10 | +0.10 |
| 2 | 0.50 | 0.55 | +0.05 |
| 3 | 1.00 | 1.00 | 0.00 |
| 4 | 0.20 | 0.25 | +0.05 |
| 5 | 0.00 | 0.00 | 0.00 |
| 6 | 0.30 | 0.35 | +0.05 |
| mean | 0.333 | 0.375 | +0.042 |

The two columns overlap almost completely: users range from 0 to 1 whatever the method. Two separate intervals, one
for A and one for B, would overlap, which suggests "no difference". But **every** user is better or equal under B.
Looking at the differences removes the users' own spread: easy users stay easy and hard users stay hard. Only the
change remains. This is why the lab judges with paired differences; the
[comparing primer](../labs/primer-comparing.md) works through the bootstrap by hand.

## The verdict and the promotion rule

Over the five datasets, the verdicts combine into one decision:

| Verdicts | Decision |
|---|---|
| better on all five | promote as the **new default** (and add it to the search space) |
| better on some, worse on none | promote as a **search-space option**: the tuner may pick it where it helps; the default stays |
| worse on any | do not promote; record what you learned |
| no clear difference anywhere | not promoted |

Last.fm (723 test users) and MovieLens (1,862) are smaller than the other three (10,000 test users each), so their
intervals are wider and "no clear difference" is common there. That is the reason the rule only asks for "worse on
none" to add an option.

## Where did it get better?

```bash
python -m recbench.compare most_popular rp3beta --datasets movielens-25m --segments
```

`--segments` splits the users into four equal groups (quartiles) three ways and repeats the paired test within each
group:

!!! success "You should see (shortened)"
    ```text
    == movielens-25m: where B - A comes from

      1-107 events before the test                            466 users  A 0.1025  B 0.1245  B - A +0.0220 [+0.0098, +0.0328]  better
      ...
      568-1000 events before the test                         466 users  A 0.2443  B 0.2933  B - A +0.0490 [+0.0371, +0.0603]  better

      0.0-14.2 days since the last event                      466 users  A 0.2108  B 0.2332  B - A +0.0224 [+0.0110, +0.0342]  better
      14.2-89.3 days since the last event                     465 users  A 0.1901  B 0.1971  B - A +0.0070 [-0.0038, +0.0171]  no clear difference
      ...
      375.0-5521.8 days since the last event                  466 users  A 0.1308  B 0.1885  B - A +0.0577 [+0.0453, +0.0698]  better

      recall@10, popular head (top 20% of items)      75507 pairs  A 0.0395  B 0.0433  B - A +0.0038
      recall@10, long tail                            12850 pairs  A 0.0000  B 0.0002  B - A +0.0002
      recall@10, new (no history before the test)     17740 pairs  A 0.0000  B 0.0000  B - A +0.0000

      most often found by B, missed by A:
         259 users  Arrival (2016) (164179) (popularity 728)
         ...
      most often found by A, missed by B:
         240 users  Thor: Ragnarok (2017) (122916) (popularity 109)
         ...
    ```

| Group | What it asks |
|---|---|
| events before the test | does the change help users with little history, or the heavy users? |
| days since the last event | does it help users who come back after a long break? |
| popularity percentile of the test items | does it help users who choose popular items, or niche ones? |
| recall@10 by test item | which kinds of items it finds more often: the popular head, the long tail, brand-new items |
| found by B, missed by A | the concrete items behind the difference |

**Reading this example.** RP3beta gains everywhere, most for heavy users and for users returning after more than a
year. MostPopular still wins on recent hits such as *Thor: Ragnarok*: only 109 ratings in all of history, but many
in the last weeks. RP3beta's popularity is all-time, so it misses them. That suggests an idea: give RP3beta a sense
of *recent* popularity ([lab 2](../labs/02-rp3beta.md), Level 3). Also, 17% of the test items had no history before
the test. No collaborative method can recommend them, whatever its settings.

Segments use the users' saved top-10 lists (the lab's `eval.save_topk`), so they work for every lab run.

## The scoreboard

```bash
python -m recbench.lab scoreboard --docs
```

It rebuilds the [scoreboard](../labs/scoreboard.md): every lab method's baseline per dataset and, once you have
experiments, the best one (chosen by **validation** score), its paired difference to the baseline, and its verdict.
`--docs` also writes `docs/generated/lab/*.md`, which the docs site shows. Commit those files in your pull request.

## Three traps

- **Trying many things and keeping the one that looks best on test.** With 20 experiments, one may look "better" by
  luck. The scoreboard therefore chooses by validation score, and the promotion rule asks for agreement across
  datasets.
- **Comparing numbers from different splits.** A validation score next to a test score, or a laptop result next to
  the box's: different users, items or item caps. Only pairs from `compare` are like for like.
- **Reading "no clear difference" as "equal".** It means *not shown*: the data cannot tell them apart. On Last.fm
  that is often the case.
