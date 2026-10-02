# Efficiency

How much time and memory a method needs, from training to answering a request. These numbers depend on the
hardware, so compare them only within one hardware profile (the `hardware` tag of each run).

## Training time: `train_seconds`

Wall-clock time of `fit()` on the active hardware. Data loading done inside `fit` is included; evaluation is
not. recbench trains for a fixed number of steps, so this measures the cost *of the budget*. A method that
needs ten times more steps to converge would take ten times longer.

## Batch scoring speed: `score_seconds_per_1k_users`

Time to score 1,000 users against the **whole catalog** during evaluation. It reflects how a method would
precompute recommendations offline:

- dot-product models (MF, EASE, SASRec): fast, one matrix product;
- pointwise rankers (DCN-V2, DIN): every (user, item) pair goes through a network, so they are orders of
  magnitude slower (see [retrieval and ranking](../concepts/retrieval-and-ranking.md)).

## Memory: `peak_rss_mb` and `peak_gpu_mb`

- `peak_rss_mb`: the peak resident memory of the process that trained and evaluated the method. Each
  (dataset, method) pair runs in **its own child process**, so the number belongs to that method alone (in
  recbench v0.1, memory accumulated across methods). macOS reports this in bytes and Linux in kilobytes;
  recbench converts both (`src/recbench/runner.py::_peak_rss_mb`).
- `peak_gpu_mb`: the peak GPU memory allocated by PyTorch (0 on CPU).

## Serving latency and throughput

Measured against a **running API** by the load tester (`src/recbench/serving/latency.py::measure`):

| Metric | Meaning |
|---|---|
| `served_p50_ms`, `served_p95_ms`, `served_p99_ms` | latency percentiles of `POST /recommend` |
| `served_rps` | requests per second achieved at the chosen concurrency |

The tester sends 20 warm-up requests first, mixes real user ids with ~10% unknown users (popularity fallback),
and runs requests concurrently. Run against Cloud Run from your laptop, latency includes the network round trip.
See [serving and latency](../concepts/serving-and-latency.md) for how to read percentiles, and
[load-test the API](../../how-to/load-test.md) for the command.

## When these mislead

- **Different machines:** a laptop and a GPU server are not comparable. Check the `hardware` tag.
- **Training time vs quality:** a fast method trained too briefly may simply be under-trained.
- **Cold starts:** the first request after idle on a serverless platform is much slower. Warm-ups exclude it on
  purpose; report it separately if it matters.

## Check your understanding

??? question "Why is DCN-V2's batch scoring much slower than iALS's?"
    iALS computes all scores with one matrix product. DCN-V2 runs a small network for every (user, item)
    pair: users × items forward passes.

??? question "Why does recbench run each method in a separate process?"
    To measure peak memory per method, to stop one crash or hang from killing the whole benchmark, and to
    free GPU memory between methods.

## Further reading

- [Cost](cost.md) and [time to market](time-to-market.md).
