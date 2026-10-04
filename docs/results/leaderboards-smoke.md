# Smoke-tier leaderboards

The tables below are **generated** from MLflow by

```bash
python -m recbench.report.build --tier smoke --tuning defaults --out reports/smoke-latest --docs
```

Never edit them by hand: re-run the command after new runs. To read them correctly, start with
[how to read results](how-to-read-results.md).

!!! warning "Smoke tier: a pipeline check, not a verdict"
    These are laptop runs on user-sampled slices (about 50,000 events per dataset, a few hundred to a few
    thousand evaluation users), with small untuned models. They show that every method runs end to end, and
    they teach you how to read a leaderboard. Many methods are statistically tied (≈). Choose methods with the
    tuned [quick-tier bake-off](quick-tier.md) and its full-data confirmations, and see which are good everywhere
    on the [overall comparison](overall-comparison.md).

The full-tier results are on the [leaderboards](leaderboards.md) page.

## MovieLens-25M (movies)

Explicit ratings treated as implicit "watched" events; long histories. See the
[dataset page](../dictionary/datasets/movielens-25m.md).

--8<-- "generated/leaderboards/smoke/movielens-25m.md"

## RetailRocket (e-commerce)

Views, add-to-carts, and purchases from an online shop; very sparse, many short-lived users. See the
[dataset page](../dictionary/datasets/retailrocket.md).

--8<-- "generated/leaderboards/smoke/retailrocket.md"

## H&M (fashion e-commerce)

Purchases with rich item metadata and product images. See the [dataset page](../dictionary/datasets/hm.md).

--8<-- "generated/leaderboards/smoke/hm.md"

## Last.fm-1K (music)

Listening events with many repeats; also evaluated with repeats allowed. See the
[dataset page](../dictionary/datasets/lastfm.md).

--8<-- "generated/leaderboards/smoke/lastfm.md"

## Steam (games)

Game reviews as implicit feedback, with game metadata. See the [dataset page](../dictionary/datasets/steam.md).

--8<-- "generated/leaderboards/smoke/steam.md"
