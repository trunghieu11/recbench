# Serving and bundles

## Why bundles

Serving a model online needs its code, its weights, and usually PyTorch: a large container that starts slowly.
For most recommendation surfaces, lists computed ahead of time are good enough. recbench therefore exports a
**bundle** per (dataset, tier, method) right after evaluation, and the API only reads files.

Trade-off: lookups are fast and cheap, but lists are only as fresh as the last export (they reflect events
before the cutoff). See [serving and latency](../dictionary/concepts/serving-and-latency.md).

## Export

`src/recbench/serving/bundle.py::export_bundle`, called by the runner for ranked, non-managed methods:

1. Pick up to 20,000 warm users, the most recently active first (`bundle_users`).
2. Score them with the method (`full_scores`), in batches of 512.
3. Remove items each user already interacted with, padding, and (for ID models) cold items, as the evaluator does.
4. Keep the top 100 (`bundle_k`) items and their scores.
5. Compute a popularity fallback list (recent popularity, all-time count as a tie-breaker).
6. Write to a temporary folder and rename it into place, so a half-written bundle is never served.

Folder: `data/bundles/<dataset>/<tier>/<method>/`

| File | Contents |
|---|---|
| `manifest.json` | dataset, tier, method, protocol version, split hash, k, user and item counts, export time, freshness note |
| `users.parquet` | user_id → row number |
| `topk.npy` | int32 [users, 100]: item indices, best first |
| `scores.npy` | float32 [users, 100]: the model's scores |
| `items.parquet` | item index → item id and display text (returned as `title`) |
| `popular.npy` | int32 [100]: the fallback list |

A bundle is at most about 16 MB of lists (20,000 users × 100 items × 8 bytes) plus the item table, so it uploads
quickly to Cloud Storage.

## The API

`src/recbench/serving/app.py` (FastAPI). Bundles are loaded on first use and cached (`load_bundle`).

| Endpoint | Purpose |
|---|---|
| `GET /health` | `{"ok": true, "version": ..., "bundles": n}` |
| `GET /methods` | available (dataset, tier, method) bundles |
| `POST /recommend` | body `{"dataset", "method", "user_id", "k" (1–100)}` → recommendations |
| `GET /dashboard` | leaderboards from MLflow (needs the `bench` extra) |
| `GET /docs` | automatic interactive API documentation |

Response of `POST /recommend`:

```json
{
  "dataset": "movielens-25m",
  "method": "ease",
  "user_id": "123",
  "fallback": false,
  "recommendations": [{"item_id": "2571", "title": "Matrix, The (1999)", "score": 0.84}]
}
```

Unknown users get the popularity list, `"fallback": true`, and `null` scores. Unknown bundles return 404.

## Why the container has no PyTorch

The API imports only NumPy, pandas, PyArrow, FastAPI, and Jinja2. The Docker image installs the `serve` extra
only. `tests/test_runner_serving.py::test_serving_does_not_import_torch` checks this. Smaller images mean faster
cold starts on Cloud Run.

## Guarantees and their tests

| Guarantee | Test |
|---|---|
| the API returns exactly the evaluator's top-K (seen items removed) | `test_bundle_matches_the_evaluator_and_falls_back_for_unknown_users` |
| unknown users get the fallback list | same test |
| the API serves, lists, and 404s correctly | `test_api_serves_bundles` |
| no torch import in the serving path | `test_serving_does_not_import_torch` |

(All in `tests/test_runner_serving.py`.)

## Real-time serving (not implemented)

A real-time variant would load the model and score the user's *current* history per request, using an ANN index
for embedding models. It is a possible extension on the [roadmap](../results/roadmap.md).
