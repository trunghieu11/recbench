# RetailRocket

> Four and a half months of browsing on a real e-commerce site: 2.76 million view, add-to-cart, and purchase
> events from 1.4 million visitors on 235,061 products.

--8<-- "generated/datasets/retailrocket.md"

## What one row means

Visitor 257597 viewed (or added to cart, or bought) product 355908 at a given millisecond. Visitors are
anonymous browser identities, not logged-in customers.

## Fields used

| Source file | Field | recbench column |
|---|---|---|
| `events.csv` | timestamp (ms), visitorid, event, itemid | timestamp, user_id, value, item_id |
| `item_properties_part1/2.csv` | rows with property "categoryid" | item category (first value found) |

Event types become values: view = 1, add-to-cart = 2, transaction = 3. Every event counts as an interaction.

## How recbench prepares it

- Cleaning: `src/recbench/datasets/retailrocket.py::Retailrocket` (adapter version 2). Item text is left empty:
  the public dump hashes all product properties except the category id.
- Split rule: the last 10% of events by time are the test window.
- Repeat policy: `exclude_seen`; repeated views of the same product are common.

## Pitfalls

- **Extreme sparsity and cold users.** Most visitors appear in a single session, so many test users have no
  history (the cold-user slice) and the warm users have very short histories.
- **Bots** are not filtered: a few visitors have implausibly many events.
- **Property snapshots:** category values come from property tables recorded over the same period, sometimes
  after the event.
- **No text:** content methods have only category ids to work with.

## Good for

Studying sparsity, cold start, and session-like behaviour; checking that methods degrade gracefully.

## How to get it

It needs a Kaggle account and an API token:

1. Create a token at <https://www.kaggle.com/settings> → "Create New Token", which downloads `kaggle.json`.
2. `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
3. `python -m recbench.pipeline.prepare --config configs/benchmarks/smoke-cpu.yaml --datasets retailrocket`

Licence: CC BY-NC-SA 4.0 (attribution, non-commercial, share-alike). Source:
<https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset>.
