# Leaderboards

These are the **full-tier** results: every dataset at full size, up to 10,000 warm evaluation users each. The tables
are generated from MLflow by

```bash
python -m recbench.report.build --tier full --tuning defaults --out reports/full-untuned-v0.2 --docs
```

Never edit them by hand. To read them correctly, start with [how to read results](how-to-read-results.md). The
small laptop runs are on the [smoke-tier page](leaderboards-smoke.md).

!!! warning "Untuned v0.2 defaults"
    These runs used **one set of default settings for every dataset**, with no tuning:

    - the `48gb` preset (dimension 128, learning rate 0.001, 30,000 training steps, batch 512);
    - no validation-based early stopping;
    - run once, on a rented 48 GB GPU machine, on 2026-10-03 and 04.

    Simple methods with one or two settings (MostPopular, EASE, ItemKNN) are less sensitive to this. Neural
    models depend heavily on their settings, so these tables understate them. Some heavy methods did not
    finish (see "Did not run" in each section). The [quick-tier bake-off](../dictionary/concepts/fair-baselines-and-tuning.md)
    re-runs the low-budget methods with equal tuning for every method.

!!! info "How long is the test window?"
    For H&M, the test window is the last 7 days. For the other datasets, the cutoff is placed so that the last 10% of
    all events form the test window. That is about two weeks for RetailRocket, two and a half months for Steam, almost
    two years for MovieLens, and several years (a thin tail) for Last.fm. "Test window starts" in each split box
    gives the exact date.

The datasets appear in the order used by the quick-tier bake-off: the most studied first, then the contrasting
regimes, the flagship shop, and the statistically weakest last.

## MovieLens-25M (movies)

Ratings treated as implicit "watched" events; dense, long histories. See the
[dataset page](../dictionary/datasets/movielens-25m.md).

--8<-- "generated/leaderboards/full/movielens-25m.md"

## RetailRocket (e-commerce)

Views, add-to-carts, and purchases; very sparse, many short-lived users. See the
[dataset page](../dictionary/datasets/retailrocket.md).

--8<-- "generated/leaderboards/full/retailrocket.md"

## Steam (games)

Game reviews as implicit feedback, with game metadata; dates only. See the
[dataset page](../dictionary/datasets/steam.md).

--8<-- "generated/leaderboards/full/steam.md"

## H&M (fashion e-commerce)

Purchases with rich item metadata; dates only. See the [dataset page](../dictionary/datasets/hm.md).

--8<-- "generated/leaderboards/full/hm.md"

## Last.fm-1K (music)

Listening events with many repeats; also evaluated with repeats allowed. See the
[dataset page](../dictionary/datasets/lastfm.md).

--8<-- "generated/leaderboards/full/lastfm.md"
