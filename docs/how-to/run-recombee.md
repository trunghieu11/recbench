# Benchmark Recombee

Compare a managed recommendation service with local methods on the same data. See the
[Recombee dictionary page](../dictionary/algorithms/recombee.md) for how the adapter works.

## 1. Create a dedicated database

1. Sign up at <https://www.recombee.com> (a free plan exists; confirm its current limits on their pricing page).
2. In the admin UI, create a **new, empty database** used only for benchmarking. recbench refuses to upload into
   a database that already contains items, unless you allow resetting it.
3. Note the **database id**, its **private token**, and the **region** (for example eu-west).

## 2. Set credentials (never commit them)

```bash
export RECBENCH_RECOMBEE_DB="your-db-id"
export RECBENCH_RECOMBEE_TOKEN="your-private-token"
export RECBENCH_RECOMBEE_REGION="eu-west"          # ap-se, ca-east, eu-west, or us-west
export RECBENCH_RECOMBEE_ALLOW_RESET=1             # allow wiping THIS database before and after the run
```

Tip: put these lines in a `.env` file (git-ignored) and run `set -a; source .env; set +a`.

## 3. Install the SDK and prepare the slice

```bash
uv pip install -e ".[managed]"
python -m recbench.pipeline.prepare --config configs/benchmarks/recombee-slice.yaml
```

The `slice` tier is sized for a free plan: about 40,000 events, at most 10,000 items, and at most 1,000
evaluation users.

## 4. Run

```bash
python -m recbench.runner --config configs/benchmarks/recombee-slice.yaml
python -m recbench.report.build --tier slice --out reports/slice-latest
```

The config runs Recombee **and** six local methods (Random, MostPopular, ItemKNN, EASE, iALS, SASRec) on the same
slice, so the comparison is fair.

What to expect:

- Before sending anything, recbench computes the number of requests the run needs and refuses if it exceeds
  `recombee_max_requests` (90,000).
- Upload takes a few minutes. Then recbench polls (up to 20 times, every 30 seconds) until the service returns
  recommendations.
- The run logs `recombee_requests_used` and live latency (`served_p50_ms`, `served_p95_ms`, `served_p99_ms`) to MLflow.

## 5. Clean up

With `RECBENCH_RECOMBEE_ALLOW_RESET=1`, the database is reset at the end (set `RECBENCH_RECOMBEE_KEEP=1` to keep the
data for inspection in the admin UI). Otherwise, delete the data in the admin UI.

## Common errors

| Message | Fix |
|---|---|
| `set RECBENCH_RECOMBEE_DB and RECBENCH_RECOMBEE_TOKEN` | export the variables (step 2) |
| `the Recombee database is not empty` | use a fresh database, or set `RECBENCH_RECOMBEE_ALLOW_RESET=1` |
| `Recombee would need ~N requests` | use the slice tier, or raise `recombee_max_requests` if your plan allows it |
| `did not return recommendations in time` | try again later; the service trains in the background |
| `recombee-api-client is not installed` | `uv pip install -e ".[managed]"` |

## Test without an account

`pytest -q tests/test_recombee.py` runs the adapter against a fake client. A live test runs only with
`RECBENCH_LIVE=1` and the credentials above.
