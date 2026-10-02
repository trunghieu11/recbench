# Time to market

## The question

How long does it take to go from raw interaction data to a working recommendation API? For a new product,
this can matter more than a few points of accuracy.

## What recbench measures automatically

`python -m recbench.ttm --method ease` runs the whole path on a small synthetic interaction log (no downloads, so
it runs anywhere) and times each stage:

| Field | Stage |
|---|---|
| `raw_to_split_s` | clean the log and build the split |
| `fit_eval_s` | fit the method and evaluate it |
| `export_s` | write the serving bundle |
| `start_to_first_s` | start the API until the first **successful** `POST /recommend` |
| `time_to_endpoint_s` | the total |

On a laptop the whole path takes a few seconds for EASE. The result is written to `runs/ttm/ttm.json`. Code:
`src/recbench/ttm.py::run`. The shortcut `./scripts/time_to_endpoint.sh` does the same, and
`METHOD=sasrec ./scripts/time_to_endpoint.sh` times another method.

Success means a real recommendation response, not just a healthy server. recbench v0.1 only waited for
`/health`, which passed even when the model could not load.

## What it does not capture

The human part usually dominates:

- writing a dataset adapter and checking data quality;
- implementing or integrating the method;
- tuning it;
- setting up cloud resources, monitoring, and on-call.

These are scored with the [qualitative rubric](qualitative-rubric.md): implementation effort and tuning effort
are the main ingredients.

## Combining both

A practical estimate of time to market for a method:

1. The automated pipeline time (from `recbench.ttm` on your real data size), plus
2. the engineering time implied by the rubric: for example, EASE (implementation 2, tuning 1) takes days,
   while HSTU (implementation 4, tuning 4) takes weeks, plus GPU access.

For managed services, the automated part is upload time plus the service's training delay, and the human part
is account setup and integration.

## Check your understanding

??? question "Why measure until the first successful recommendation rather than until the server starts?"
    A server can be healthy and still fail on every real request (for example, if a model cannot load). Only a
    real response proves the path works.

??? question "Why is the toy log used instead of a real dataset?"
    It makes the measurement fast, free, and identical on every machine, so stages can be compared. For real
    sizes, run the benchmark and read `train_seconds`.
