# Other managed services

> Cloud providers also rent recommenders. recbench documents three of them: Amazon Personalize, Google's
> commerce recommender, and Azure AI Personalizer, which retired on 2026-10-01. Only [Recombee](recombee.md)
> is benchmarked live; this page explains why, and what each service offers.

--8<-- "generated/services.md"

## Amazon Personalize (AWS)

**What it is.** A fully managed service built on the technology behind Amazon.com recommendations. You
import interactions (and optionally user and item metadata) into a *dataset group*, train a *solution* with
a *recipe*, and deploy it as a *campaign* that answers real-time requests. Batch jobs can score many users
at once.

**Recipes (model types)** include:

| Recipe | Purpose |
|---|---|
| User-Personalization-v2 | "recommended for you" lists |
| Personalized-Ranking-v2 | re-rank a list you provide (a second-stage ranker) |
| Similar-Items | "more like this" for an item page |
| Trending-Now / Popularity-Count | popularity-style baselines |

**Business rules.** Filters (for example "only in-stock items") and promotions are part of the API.

**Why recbench does not benchmark it live:**

1. **Cost model.** Real-time campaigns are billed for provisioned throughput per hour, even while idle, plus
   training hours. One forgotten campaign can exceed the project's <$50/month budget. Batch inference
   avoids the idle cost but still bills training.
2. **Authentication.** Requests must be signed with AWS Signature Version 4, normally through `boto3`. The
   v0.1 adapter sent unsigned requests, which can never work.

**Status:** available (it is not on AWS's list of services in maintenance as of 2026-10-02).
Pricing: <https://aws.amazon.com/personalize/pricing/>.

## Google Cloud: Vertex AI Search for commerce

**What it is.** Google's retail recommendation and search service (formerly Recommendations AI and the
Retail API; newer documentation also calls it "AI Commerce Search"). It needs a **product catalog** and
**user events** (views, add-to-carts, purchases), and offers model types such as "Others you may like",
"Frequently bought together", and "Recommended for you". Predictions are served through a placement or
serving configuration.

**Why recbench does not benchmark it live:**

1. **Retail-only.** It expects a product catalog, so movies, music, and games do not fit naturally.
2. **Setup and training time.** Catalog import, event import, and model training take hours to days before
   the first prediction.
3. **Cost and access.** Training and predictions are billed, and access requires a configured Google Cloud
   project.

Pricing: <https://cloud.google.com/retail/pricing>.

## Azure AI Personalizer (retired)

**What it was.** A *contextual bandit* service: for each request, you sent a context (user and situation
features) and a short list of actions (items), and it chose one action, learning from the reward you reported
back (reinforcement learning). It did not learn from interaction histories the way collaborative filtering does.

**Status:** Microsoft stopped allowing new Personalizer resources on 2023-09-20 and **retired the service on
2026-10-01**. Existing deployments stopped working on that date. Microsoft's guidance points to building
custom solutions, for example with Azure Machine Learning. Documentation (archived):
<https://learn.microsoft.com/en-us/azure/ai-services/personalizer/>.

## Build or buy?

| Question | Managed service | Build (recbench methods) |
|---|---|---|
| Time to a first working API | hours to days | days to weeks |
| Control over ranking logic | settings and rules only | full |
| Explanations | usually none | exact (ItemKNN, EASE) or post-hoc |
| Data leaves your infrastructure | yes | no |
| Cost at small scale | free tiers or tens of dollars per month | your compute time |
| Cost at large scale | grows with users, items, and requests | grows with compute and engineers |
| Lock-in | high (APIs, data formats) | low |

recbench's [decision guide](../../results/decision-guide.md) combines this table with benchmark results.

## Adding another service

Follow `src/recbench/methods/recombee.py::RecombeeMethod`:

1. Use the provider's official SDK (it handles authentication).
2. Check the request or cost budget *before* sending anything.
3. Upload only `TrainView.export_frame()`, the pre-test events.
4. Implement `topk` (set `output="list"` in the spec), so the evaluator handles seen items and metrics.
5. Add teardown, and a `finish()` that reports latency and usage.
6. Add a catalog entry and a page in this dictionary.

## Check your understanding

??? question "Why is an idle real-time campaign a cost risk, but a batch job is not?"
    A campaign reserves serving capacity and is billed per hour whether or not it receives requests. A batch
    job is billed only while it runs.

??? question "Why is a contextual bandit not a drop-in replacement for collaborative filtering?"
    It picks one action from a short list you provide for each context, learning from rewards. It does not
    learn item-to-item or user-to-item patterns from histories, so you must still generate the candidates.
