# Serving and latency

## Why it matters

A model is only useful once it answers requests fast enough. Users notice delays of a few hundred
milliseconds. Serving choices also drive cost: an always-on GPU server costs money every hour, while
precomputed lists cost almost nothing to serve.

## Two ways to serve

| | Batch (precomputed) | Real-time (online) |
|---|---|---|
| How | compute every user's top-K ahead of time; store; look up on request | run the model when the request arrives |
| Latency | very low (a lookup) | model-dependent |
| Freshness | as old as the last batch | reacts to the latest action |
| Cost | cheap to serve; compute once per batch | servers sized for peak traffic |
| Fits | stable tastes, daily updates | sessions, news, "up next" |

**recbench serves batch-style bundles**: after evaluation, each method exports the top 100 items for up to
20,000 recently active users, plus a popularity list for unknown users. The API reads these files, needs no
PyTorch, and answers in about a millisecond plus network time. See
[serving and bundles](../../codebase/serving-and-bundles.md).

For large catalogs with embedding models, real-time systems use **approximate nearest-neighbour (ANN)**
indexes (for example FAISS or HNSW) to find the highest dot products without scoring every item. recbench does
not need one yet, because bundles are precomputed.

## Measuring latency: percentiles

Latency varies from request to request. Averages hide the slow ones, so systems report **percentiles**:

- **p50 (median):** half of the requests were faster.
- **p95:** 95% were faster; 1 in 20 users waited longer.
- **p99:** 1 in 100 waited longer, often the users who notice.

Example: latencies (ms) 12, 13, 13, 14, 15, 15, 16, 18, 25, 250. The median is 15 ms. One slow request
(250 ms) barely moves the median, but dominates p99. That is why recbench reports p50, p95, and p99.

**Throughput** (requests per second, RPS) is the other half: how many requests the service handles with a
given number of requests in flight (concurrency).

## Cold starts on serverless platforms

Cloud Run scales to zero instances when idle (cheap). The first request after idle must start a container: a
**cold start** of a second or more. The bundle-based image is small (no PyTorch), which keeps cold starts short.
The load tester sends warm-up requests first, so they do not distort the percentiles.

## In recbench

- API: `src/recbench/serving/app.py` (`POST /recommend`); bundles: `src/recbench/serving/bundle.py::Bundle`.
- Load test: `python -m recbench.serving.latency --base-url ... --dataset ... --method ...` reports
  `served_p50_ms`, `served_p95_ms`, `served_p99_ms`, and `served_rps`, and attaches them to the method's MLflow run.
- Model-side speed: `score_seconds_per_1k_users`, measured during evaluation.

## Pitfalls

- **Reporting the mean** instead of percentiles.
- **Measuring from far away:** latency from your laptop to a cloud region includes the network round trip.
- **Including cold starts** in steady-state numbers, or forgetting they exist.

## Check your understanding

??? question "In the example, why does p99 differ so much from p50?"
    One request out of ten took 250 ms. Percentiles near the top are dominated by such outliers.

??? question "Why can bundle-based serving not react to a click the user made a minute ago?"
    The lists were computed at export time, from history up to the cutoff. New events reach users only after
    the next export.

## Further reading

- [Efficiency metrics](../metrics/efficiency.md) and [Cloud Run](../../cloud/cloud-run.md).
- FAISS: <https://github.com/facebookresearch/faiss>.
