# Load-test the API

Measure latency percentiles and throughput of a running API, and attach them to the method's MLflow run.

## Run it

Against a local API ([serve locally](serve-locally.md)):

```bash
python -m recbench.serving.latency --base-url http://127.0.0.1:8080 --dataset movielens-25m --method ease
```

Against a private Cloud Run service:

```bash
python -m recbench.serving.latency --base-url "$URL" --token "$(gcloud auth print-identity-token)" \
       --dataset movielens-25m --method ease --requests 500 --concurrency 8
```

Options: `--requests` (default 300), `--concurrency` (requests in flight, default 8), `--tier` (default smoke),
and `--no-log` (do not write to MLflow).

## What it does

1. Picks real user ids from the bundle, plus about 10% unknown ids (which exercise the popularity fallback).
2. Sends 20 warm-up requests (so cold starts do not count).
3. Sends the measured requests concurrently, timing each one.
4. Prints, and logs to the latest finished MLflow run of that (dataset, method):

```json
{
  "served_p50_ms": 4.1,
  "served_p95_ms": 9.8,
  "served_p99_ms": 15.2,
  "served_rps": 1650.3,
  "served_requests": 300.0,
  "served_concurrency": 8.0,
  "logged": 1.0
}
```

(Illustrative values for a local API.)

## Reading the numbers

- **p50** is the typical request; **p95/p99** are what your unluckiest users see. Optimise the tail.
- From your laptop to Cloud Run, most of the time is the **network round trip**: the same bundle can answer in a
  few milliseconds locally and in 50–200 ms remotely, depending on the region.
- **RPS** depends on concurrency. Raise `--concurrency` until p95 starts to grow: that is the service's practical capacity.
- On Cloud Run with `max-instances 1`, throughput is capped by one instance on purpose (cost safety).

See [serving and latency](../dictionary/concepts/serving-and-latency.md) and [efficiency](../dictionary/metrics/efficiency.md).
