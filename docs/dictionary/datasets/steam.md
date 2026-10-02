# Steam

> 7.8 million reviews of 15,474 games by 2.57 million Steam users (2010–2018), collected by the McAuley Lab at
> UC San Diego, plus game metadata (titles, tags, genres).

--8<-- "generated/datasets/steam.md"

## What one row means

User "Chaos Syren" wrote a review of game 725280 on 2017-12-17. recbench treats a review as an interaction with
the game; the hours played are kept as `value`.

## Fields used

| Source file | Field | recbench column |
|---|---|---|
| `steam_reviews.json.gz` | username, product_id, date, hours | user_id, item_id, timestamp, value |
| `steam_games.json.gz` | title + up to 15 tags | item text |
| `steam_games.json.gz` | genres | item category tokens |

## How recbench prepares it

- Cleaning: `src/recbench/datasets/steam.py::Steam` (adapter version 2). The files contain Python-style
  dictionaries rather than JSON, so recbench extracts the four needed fields with a fast regular expression
  and falls back to `ast.literal_eval` for unusual lines. All 7.8 million reviews are read (v0.1 stopped at
  5 million, which biased the data by file order).
- **Item text comes from game metadata, never from reviews:** a review can be written after the test cutoff,
  and using it as item content would leak the future (a v0.1 bug).
- Split rule: the last 10% of reviews by time are the test window. Dates are day-level, so ties are ordered by
  file position.
- Repeat policy: `exclude_seen`.

## Pitfalls

- **Reviews are not plays:** people review a small, biased subset of what they play.
- **Popularity-heavy:** a few blockbuster games collect most reviews, so MostPopular is hard to beat.
- **Usernames are identities:** two users with the same display name would be merged (rare).

## Good for

Popularity bias, content features from tags and genres, and a small catalog with many users.

## How to get it

No account needed. recbench downloads both files from <https://mcauleylab.ucsd.edu/public_datasets/data/steam/>:

```bash
python -m recbench.pipeline.prepare --config configs/benchmarks/smoke-cpu.yaml --datasets steam
```

The games metadata file is optional: without it, items have no text or categories. Licence: provided by the
McAuley Lab for research purposes. Cite Kang & McAuley (2018),
[Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781).
