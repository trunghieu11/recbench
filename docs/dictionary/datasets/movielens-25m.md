# MovieLens 25M

> 25 million movie ratings from 162,541 users on 59,047 movies, collected by the GroupLens research lab
> from its movie recommendation site.

--8<-- "generated/datasets/movielens-25m.md"

## What one row means

User 123 rated movie 456 with 4.0 stars at a given time. recbench treats every rating, high or low, as
"user 123 watched movie 456" (an implicit interaction). The star value is kept in the `value` column but not
used for ranking.

## Fields used

| Source file | Field | recbench column |
|---|---|---|
| `ratings.csv` | userId, movieId, rating, timestamp (seconds) | user_id, item_id, value, timestamp |
| `movies.csv` | title | item text |
| `movies.csv` | genres, e.g. "Adventure\|Animation\|Comedy" | item category tokens ("(no genres listed)" becomes empty) |

## How recbench prepares it

- Cleaning: `src/recbench/datasets/movielens.py::MovieLens25M` (adapter version 2).
- Split rule: the last 10% of all ratings by time are the test window; the 10% before that are the validation window.
- Repeat policy: `exclude_seen` (re-ratings are rare).
- Ratings span 1995 to 2019, so the test window covers the most recent period.

## Pitfalls

- **Timestamps are rating times, not viewing times.** Many users rate dozens of movies in one sitting when they
  join, so the order of ratings says little about the order of viewing. Sequential models have weaker signal
  here than the dataset's size suggests.
- **Survivorship:** every user in MovieLens rated at least 20 movies, so very light users are missing.
- **It is famous:** language models may have memorised facts about it, a concern for LLM-based methods.

## Good for

Classic collaborative filtering comparisons, genre-based diversity and calibration metrics, and teaching. Less
good for sequential claims.

## How to get it

No account needed. recbench downloads `https://files.grouplens.org/datasets/movielens/ml-25m.zip` (about 250 MB):

```bash
python -m recbench.pipeline.prepare --config configs/benchmarks/smoke-cpu.yaml --datasets movielens-25m
```

Licence: GroupLens usage terms. Research use; do not redistribute; ask permission for commercial use. Cite
Harper & Konstan (2015), [The MovieLens Datasets: History and Context](https://dl.acm.org/doi/10.1145/2827872).
