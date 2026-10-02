# Cost

## The question

What does it cost to train, serve, and maintain a method? Cost often decides between methods whose accuracy is
statistically tied.

## How to think about it

Total cost of ownership has four parts:

| Part | Driven by | Example |
|---|---|---|
| Training compute | training time × price of the machine × how often you retrain | a GPU for 2 hours daily |
| Serving compute | requests × compute per request, plus idle capacity | an always-on server vs scale-to-zero |
| Storage and data transfer | data, model files, bundles, traffic leaving the cloud | Cloud Storage bucket, egress |
| People | engineering time to build, tune, monitor, and fix | weeks for a deep model vs days for EASE |

Two patterns matter most:

- **Always-on vs scale-to-zero.** A reserved server or a managed "campaign" bills every hour, even with no
  traffic. Serverless containers (Cloud Run with `min-instances 0`) bill only while handling requests.
- **Per-request pricing.** Managed services often charge per recommendation request; cost grows linearly with traffic.

## A worked example (illustrative prices)

Suppose a GPU machine costs $1.00 per hour, and training takes 30 minutes, once a day:
30 days × 0.5 hours × $1.00 = **$15 per month** for training. Serving precomputed bundles from a scale-to-zero
container at low traffic may stay within a free tier. An always-on server at $0.05 per hour costs
24 × 30 × 0.05 = **$36 per month** even with zero users. Check your provider's current prices; they change.

## What recbench measures today

| Signal | Where |
|---|---|
| training time per method | `train_seconds` ([efficiency](efficiency.md)) |
| scoring time per 1,000 users | `score_seconds_per_1k_users` |
| memory | `peak_rss_mb`, `peak_gpu_mb` (decides the machine size) |
| serving latency and throughput | `served_*` metrics from the load test |
| managed-service usage | `recombee_requests_used` |
| a rough cost class per method | `cost_band` in the method's spec (shown in the generated facts) |

A **cost = runtime × price table** (an editable file of machine prices, turned into dollars per run) is on the
[roadmap](../../results/roadmap.md). Until then, multiply `train_seconds` by your machine's hourly price yourself.

## In recbench's setup

- The laptop runs the smoke tier for free; the full tier runs on your own GPU machine (no cloud GPU bill).
- The API runs on Cloud Run with `min-instances 0` and `max-instances 1`, so it cannot scale into a large bill.
- A budget alert script warns at 50%, 90%, and 100% of a monthly budget (alerts do not stop spending):
  [cost control](../../cloud/cost-control.md).

Official pricing pages: [Cloud Run](https://cloud.google.com/run/pricing), [GPUs on Compute
Engine](https://cloud.google.com/compute/gpus-pricing).

## Check your understanding

??? question "Why can a managed service be cheaper than building at small scale but more expensive at large scale?"
    Free tiers and no engineering work favour it early. Per-request fees grow with traffic, while your own
    serving cost grows more slowly once built.

??? question "What single setting keeps a Cloud Run service from costing money when nobody uses it?"
    `min-instances 0` (scale to zero).
