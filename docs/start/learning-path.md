# Learning path

A six-week plan for learning recommender systems with this repository, assuming Python and basic machine
learning but no recommender, deep learning, or cloud experience. Plan on about five to eight hours a week.
Each week ends with a checkpoint: if you can do it, move on.

## Week 1: the problem and the data

- Read: [Recommender systems 101](../dictionary/concepts/recsys-101.md),
  [explicit vs implicit feedback](../dictionary/concepts/feedback-types.md),
  [the interaction matrix](../dictionary/concepts/interaction-matrix.md).
- Do: [install recbench](install.md) and complete [your first smoke run](first-smoke-run.md) on MovieLens only.
- Explore: open `data/splits/movielens-25m/smoke/train.parquet` in a notebook with pandas and plot interactions
  per user and per item.
- **Checkpoint:** you can explain why 99.7% of the MovieLens matrix is empty and what "pre-test" means.

## Week 2: baselines and evaluation

- Read: [MostPopular](../dictionary/algorithms/most-popular.md), [ItemKNN](../dictionary/algorithms/itemknn.md),
  [EASE](../dictionary/algorithms/ease.md), [ranking accuracy](../dictionary/metrics/ranking-accuracy.md),
  [data leakage and time splits](../dictionary/concepts/data-leakage-and-splits.md).
- Do: [reading the results](reading-results.md). Recompute NDCG for one user by hand from
  `per_user_metrics.npz` in MLflow.
- **Checkpoint:** you can compute NDCG@5 for a toy list on paper, and say why a random split leaks.

## Week 3: learned vectors

- Read: [embeddings](../dictionary/concepts/embeddings.md), [loss functions](../dictionary/concepts/loss-functions.md),
  [negative sampling](../dictionary/concepts/negative-sampling.md), [BPR-MF](../dictionary/algorithms/bpr-mf.md),
  [iALS](../dictionary/algorithms/ials.md), [LightGCN](../dictionary/algorithms/lightgcn.md).
- Do: run `--methods bpr_mf,ials,lightgcn` on two datasets; compare coverage and popularity percentile with EASE.
- **Checkpoint:** you can explain BPR's loss and why iALS needs confidence weights.

## Week 4: sequences

- Read: [sequential and session-based](../dictionary/concepts/sequential-and-session.md),
  [SASRec](../dictionary/algorithms/sasrec.md), [BERT4Rec](../dictionary/algorithms/bert4rec.md),
  [HSTU](../dictionary/algorithms/hstu.md).
- Do: run the copy-task test (`pytest -q tests/test_methods.py -k copy`) and read it. Then compare next-item
  metrics of SASRec and EASE on Last.fm.
- **Checkpoint:** you can draw right-aligned vs left-aligned padding and say which position holds the user state.

## Week 5: beyond accuracy, and building something

- Read: [evaluation protocols](../dictionary/concepts/evaluation-protocols.md),
  [full ranking vs sampled](../dictionary/metrics/sampled-vs-full.md),
  [coverage and popularity](../dictionary/metrics/coverage-and-popularity.md),
  [confidence intervals](../dictionary/metrics/confidence-intervals.md),
  [cold start](../dictionary/concepts/cold-start.md).
- Do: [your first new method](your-first-method.md).
- **Checkpoint:** you added a method, it appears on the leaderboard, and you can say whether it is tied with
  the best method.

## Week 6: production

- Read: [retrieval and ranking](../dictionary/concepts/retrieval-and-ranking.md),
  [serving and latency](../dictionary/concepts/serving-and-latency.md), [Docker](../cloud/docker.md),
  [Cloud Run](../cloud/cloud-run.md), [cost control](../cloud/cost-control.md).
- Do: [serve locally](../how-to/serve-locally.md), then [deploy to Cloud Run](deploy-cloud-run.md), load-test
  it, and tear it down.
- **Checkpoint:** you can explain p95 latency, and you have a budget alert and a teardown command ready.

## After that

- Run the [quick-tier bake-off](quick-tier-box.md) on a rented GPU box and read its [results](../results/quick-tier.md).
- Run the [full tier on a GPU machine](full-tier-gpu.md) and compare with the smoke results.
- Benchmark a managed service: [Recombee](../how-to/run-recombee.md).
- Pick an item from the [roadmap](../results/roadmap.md) and implement it.
