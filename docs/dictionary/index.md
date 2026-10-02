# The recommender-systems dictionary

This part of the site is a reference you can come back to: one page per idea, per algorithm, per metric,
and per dataset. Every page explains the intuition first, then a tiny worked example, then the key math
symbol by symbol, then where the idea lives in recbench's code.

## How it is organised

| Section | What you will find | Start with |
|---|---|---|
| **Concepts** | the vocabulary and the traps: feedback, embeddings, losses, leakage, evaluation | [Recommender systems 101](concepts/recsys-101.md) |
| **Algorithms** | 17 methods on a ladder from Random to HSTU, plus managed services | [The method ladder](algorithms/index.md) |
| **Metrics** | how recommendations are judged: accuracy, beyond accuracy, cost, explainability | [Metrics overview](metrics/index.md) |
| **Datasets** | the five public datasets, their licences, and their quirks | [Datasets overview](datasets/index.md) |

## A reading order for beginners

1. [Recommender systems 101](concepts/recsys-101.md), [explicit vs implicit feedback](concepts/feedback-types.md),
   [the interaction matrix](concepts/interaction-matrix.md).
2. Baselines: [MostPopular](algorithms/most-popular.md), [ItemKNN](algorithms/itemknn.md), [EASE](algorithms/ease.md).
3. How results are judged: [ranking accuracy](metrics/ranking-accuracy.md),
   [evaluation protocols](concepts/evaluation-protocols.md), [data leakage](concepts/data-leakage-and-splits.md).
4. Learned vectors: [embeddings](concepts/embeddings.md), [loss functions](concepts/loss-functions.md),
   [BPR-MF](algorithms/bpr-mf.md), [iALS](algorithms/ials.md), [LightGCN](algorithms/lightgcn.md).
5. Order matters: [sequential models](concepts/sequential-and-session.md), [SASRec](algorithms/sasrec.md),
   [BERT4Rec](algorithms/bert4rec.md), [HSTU](algorithms/hstu.md).
6. Production thinking: [retrieval and ranking](concepts/retrieval-and-ranking.md), [cold start](concepts/cold-start.md),
   [serving and latency](concepts/serving-and-latency.md), [explainability](concepts/explainability.md).

## All concept pages

| Page | In one line |
|---|---|
| [Recommender systems 101](concepts/recsys-101.md) | What recommenders do and the main families of methods. |
| [Explicit vs implicit feedback](concepts/feedback-types.md) | Ratings versus clicks, and why "no click" is not "dislike". |
| [The interaction matrix](concepts/interaction-matrix.md) | Users × items, why it is 99.9% empty, and how it is stored. |
| [Embeddings](concepts/embeddings.md) | Users and items as vectors; dot products as scores. |
| [Collaborative, content-based, hybrid](concepts/collaborative-content-hybrid.md) | Learning from people versus learning from item descriptions. |
| [Retrieval and ranking](concepts/retrieval-and-ranking.md) | Why real systems use two stages. |
| [Negative sampling](concepts/negative-sampling.md) | Inventing "negatives" when you only observe positives. |
| [Loss functions](concepts/loss-functions.md) | BCE, BPR, softmax cross-entropy, sampled softmax, InfoNCE. |
| [Sequential and session-based](concepts/sequential-and-session.md) | When the order of actions carries the signal. |
| [Cold start](concepts/cold-start.md) | New users and new items. |
| [Popularity bias](concepts/popularity-bias.md) | The rich get richer, and how metrics reveal it. |
| [Data leakage and time splits](concepts/data-leakage-and-splits.md) | The most common way benchmarks lie, including recbench's own past bugs. |
| [Evaluation protocols](concepts/evaluation-protocols.md) | Full ranking versus sampled metrics, warm versus cold users, repeats. |
| [Offline vs online evaluation](concepts/offline-vs-online.md) | What offline numbers can and cannot tell you. |
| [Fair baselines and tuning](concepts/fair-baselines-and-tuning.md) | Why simple methods keep winning, and how to compare fairly. |
| [Explainability](concepts/explainability.md) | Exact versus post-hoc reasons for a recommendation. |
| [Serving and latency](concepts/serving-and-latency.md) | Batch versus real-time serving; p50, p95, p99. |
| [LLMs and generative recommenders](concepts/llm-and-generative-recsys.md) | The 2023–2026 landscape and what recbench covers. |
| [Glossary](concepts/glossary.md) | Short definitions of every term used on this site. |
