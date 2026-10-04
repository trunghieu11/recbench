# Datasets overview

recbench uses five public datasets chosen to cover different **domains**, **feedback types**, and
**data shapes**, so that a method's strengths and weaknesses show up somewhere.

| Dataset | Domain | Feedback | Side information | Shape | Commercial use |
|---|---|---|---|---|---|
| [MovieLens 25M](movielens-25m.md) | movies | ratings (used as "watched") | titles, genres | many users, medium catalog, dense | no (without permission) |
| [RetailRocket](retailrocket.md) | e-commerce | views, carts, purchases | category ids | huge, extremely sparse, many one-visit users | no (CC BY-NC-SA 4.0) |
| [H&M](hm.md) | fashion e-commerce | purchases | names, descriptions, product types, customer attributes, optional images | large, daily timestamps, test = last 7 days | no (competition rules) |
| [Last.fm 1K](lastfm.md) | music | plays | artist names | few users with very long, repetitive histories | no (non-commercial) |
| [Steam](steam.md) | video games | reviews | titles, tags, genres | many users, small catalog, popularity-heavy | no (research use) |

Full-size statistics (as cleaned by recbench): MovieLens 25,000,095 interactions; RetailRocket 2,756,101; H&M
31,788,324; Last.fm 19,150,868; Steam 7,793,069. Sparsity ranges from 89% (Last.fm) to 99.9992% (RetailRocket);
see [the interaction matrix](../concepts/interaction-matrix.md).

## What each dataset is good for

| Want to study... | Use |
|---|---|
| a classic, well-understood benchmark | MovieLens (but see its timestamp caveat) |
| extreme sparsity and cold users | RetailRocket |
| content features, images, a realistic "next week" test | H&M |
| repeat consumption and long sequences | Last.fm |
| popularity bias | Steam |

## Licensing: what you may and may not do

All five datasets are licensed for **non-commercial research** use (details on each page). recbench:

- **never redistributes** them: downloaders fetch each dataset from its original source, and `data/` is not in git;
- tags every dataset's licence in `dictionary/catalog.yaml`, shown on each page;
- is itself MIT-licensed. The *code* can be used commercially; these *datasets* cannot without permission.

If you choose a method for a commercial product, re-run the benchmark on your own data.

## Splits on your machine

Each dataset page shows the splits that exist on your machine (smoke, standard, quick, slice, full), read from their
`meta.json` files. Validation folds (`quick-val`, `full-val`, …) are left out of those tables. Create the splits with:

```bash
python -m recbench.pipeline.prepare --config configs/benchmarks/smoke-cpu.yaml --datasets movielens-25m
```

## Adding a dataset (or your own data)

See [add a dataset](../../how-to/add-a-dataset.md), including a checklist for private data.
