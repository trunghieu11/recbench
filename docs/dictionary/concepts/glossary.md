# Glossary

Short definitions of the terms used on this site. Links point to the page that explains each one in depth.

## A–C

A/B test
:   An online experiment: live traffic is split at random between two systems and their outcomes (clicks,
    purchases) are compared. See [offline vs online](offline-vs-online.md).

ALS (alternating least squares)
:   Fitting user and item vectors by solving for one side exactly while the other is fixed, then swapping.
    See [iALS](../algorithms/ials.md).

ANN (approximate nearest neighbours)
:   Index structures (FAISS, HNSW) that find vectors with the highest similarity without comparing against
    every item. See [serving and latency](serving-and-latency.md).

Attention
:   A neural mechanism where each position computes weights over other positions and mixes their information.
    See [SASRec](../algorithms/sasrec.md).

AUC (area under the ROC curve)
:   The probability that a random positive is scored above a random negative. See
    [CTR and rating metrics](../metrics/ctr-and-rating-metrics.md).

Backfill
:   In the bake-off queue: a worker that would otherwise wait starts a job of the next dataset early. Jobs of earlier datasets always come first.

Baseline
:   A simple reference method that any proposed method must beat. See [fair baselines](fair-baselines-and-tuning.md).

Bootstrap
:   Estimating uncertainty by recomputing a statistic on many resamples (with replacement) of the data. See
    [confidence intervals](../metrics/confidence-intervals.md).

BPR (Bayesian Personalized Ranking)
:   A pairwise loss: an interacted item should score above a random one. See [loss functions](loss-functions.md#bpr).

Bundle
:   recbench's precomputed top-K lists for one method and dataset, served by the API. See
    [serving and bundles](../../codebase/serving-and-bundles.md).

Candidate recall
:   For a two-stage recommender: the share of a user's test items that stage 1 put in the candidate list. No re-ranker can do better, so it is the ceiling of the second stage.

Catalog
:   The set of all items that can be recommended.

Cold start
:   Making recommendations for users or items without history. See [cold start](cold-start.md).

Collaborative filtering
:   Learning from the behaviour of many users ("people like you liked..."). See
    [collaborative, content-based, hybrid](collaborative-content-hybrid.md).

Confidence interval (CI)
:   A range that likely contains the true value of a metric. recbench reports 95% bootstrap intervals.

Confirmation
:   The bake-off's last step for a dataset: its top 3 methods by validation score are re-checked on the full data (`full-val`), then tested on `full`, with 3 seeds when training is random. It also writes their serving bundles.

Content-based
:   Using item descriptions (text, categories, images) to recommend.

Cosine similarity
:   The dot product of two vectors divided by their lengths; it compares directions, range [−1, 1].

Coverage
:   The share of the catalog that a method ever recommends. See [coverage and popularity](../metrics/coverage-and-popularity.md).

Critical difference
:   The smallest gap between two mean ranks that a statistical test (Nemenyi, Demšar 2006) accepts as real, given the number of methods and datasets. With few datasets it is large. See the [overall comparison](../../results/overall-comparison.md).

CSR (compressed sparse row)
:   A storage format keeping only the non-zero entries of a matrix. See [the interaction matrix](interaction-matrix.md).

CTR (click-through rate)
:   Clicks divided by impressions. CTR models predict the click probability of one (user, item) pair.

Cutoff
:   The time that separates training data (before) from test data (after). recbench uses one global cutoff.

## D–H

Data leakage
:   Training on information that would not be available at prediction time. See [data leakage](data-leakage-and-splits.md).

DCG / NDCG
:   Discounted cumulative gain: rewards relevant items more when they are ranked higher. NDCG divides by the best
    possible DCG. See [ranking accuracy](../metrics/ranking-accuracy.md#ndcg).

Decay (half-life)
:   A recency weight: an interaction `h` days old counts half as much as one from today, one `2h` days old a quarter, and so on (`decay_half_life_days`).

Dot product
:   The sum of element-wise products of two vectors; the usual score between a user and an item vector.

Early stopping
:   Ending training when the validation score has not improved for a number of epochs (the *patience*), and keeping the best weights seen so far.

Embedding
:   A learned vector representing a user, an item, or anything else. See [embeddings](embeddings.md).

Epoch
:   One full pass over the training data. The newer recbench methods train in epochs and stop early when the
    validation score stops improving; the older ones train for a fixed number of *steps*.

Experiment (lab)
:   A named change to how a method is tuned (settings added, removed or fixed, or a code variant switched on), run with the baseline's budget on every dataset and compared with it. See [run experiments](../../handbook/run-experiments.md).

Explainability
:   The ability to say why an item was recommended. See [explainability](explainability.md).

Explicit feedback
:   Users stating their opinion, such as star ratings. See [feedback types](feedback-types.md).

Fallback
:   What the API returns for a user it does not know: the popularity list, marked `"fallback": true`. A high fallback share means the served lists miss many callers.

Focused experiment
:   A lab experiment with `from_baseline: true`: every setting the baseline searched stays at that dataset's best value, so the trials go only to the new settings. Best for testing one idea.

Fold (validation fold)
:   A copy of a split cut one window earlier, with every real test event deleted (`quick-val`, `full-val`). Settings are tuned on it, so the test split is used only once.

Fold-in
:   Scoring a user who was not in training by passing their history through the trained model (the VAEs do this), instead of looking up a learned user vector.

Full ranking
:   Ranking the true item against the entire catalog (recbench's main protocol). See
    [full ranking vs sampled](../metrics/sampled-vs-full.md).

Gate (quick-tier)
:   The rule that every method, new or heavy, enters the comparisons through a tuned quick-tier job on every dataset.

Gini coefficient
:   A measure of inequality; 0 = everything shown equally often, 1 = one item gets all the exposure.

Golden test
:   A test that stores a method's output for fixed input (here, the toy data at default settings) and fails when it changes. `tests/test_lab_defaults.py` keeps the lab methods' defaults from changing by accident. See [test](../../handbook/test.md).

Graph filter
:   A training-free way to score items by smoothing the user-item graph with a fixed formula, as in GF-CF and Turbo-CF.

Hard negatives
:   Plausible items the user did not choose, such as a re-ranker's other candidates. They teach a model much more than random items do.

Head / long tail
:   The few very popular items (head) and the many rarely used ones (tail). See [popularity bias](popularity-bias.md).

Hit rate
:   The share of users with at least one relevant item in their top K.

Hyperparameter
:   A setting chosen before training (learning rate, embedding size), not learned from data.

## I–N

Implicit feedback
:   Behaviour signals such as clicks, plays, or purchases, from which preferences are inferred.

Interaction
:   One event connecting a user and an item (a rating, a play, a purchase).

Item
:   Anything that can be recommended: a movie, a product, an artist, a game.

Item cap
:   The limit on catalog size for models that keep a dense item × item matrix (EASE, Turbo-CF): only the most popular items are kept, so the rest cannot be recommended by them.

KL divergence
:   A measure of how different two probability distributions are. VAEs add it to their loss to keep each user's code close to a simple prior.

Label (experiment)
:   The name of a lab experiment, such as `no-time-knobs`. Its job summary is `<method>@<label>.json`, and `python -m recbench.compare ease ease:edlae` compares it with the baseline.

LambdaRank
:   A learning-to-rank objective that weights each pair of items by how much swapping them would change NDCG, so mistakes at the top of the list cost most. The LightGBM re-ranker uses it.

Latent factor
:   One dimension of a learned embedding; it has no fixed human meaning.

Logit
:   A raw model score before the sigmoid or softmax turns it into a probability.

MAP (mean average precision)
:   The average of precision values at the ranks of relevant items. See [ranking accuracy](../metrics/ranking-accuracy.md#map).

Masking
:   Hiding parts of the input (for example, future positions in attention) or removing items from a ranking
    (for example, items already seen).

MLflow
:   An experiment-tracking tool; recbench logs every run's settings and metrics to it. See [MLflow](../../cloud/mlflow.md).

MRR (mean reciprocal rank)
:   The average of 1/rank of the first relevant item. See [ranking accuracy](../metrics/ranking-accuracy.md#mrr).

Multiple comparisons
:   Running many tests at once: with a 95% interval, about one test in 20 shows a difference by chance. Agreement across datasets guards against it. See [comparing methods fairly](../../labs/primer-comparing.md).

Negative sampling
:   Picking non-interacted items as stand-in negatives. See [negative sampling](negative-sampling.md).

Next-item prediction
:   Predicting the very next item a user will interact with. See [sequential models](sequential-and-session.md).

Novelty
:   How unexpected (unpopular) recommended items are. See [novelty, diversity, serendipity](../metrics/novelty-diversity-serendipity.md).

## O–R

Over budget
:   A bake-off job status: no setting finished within the job's 3-hour cap. It is a result about cost, not a crash.

Padding
:   Filler (item 0) used to make histories of different lengths fit into one rectangular batch.

Paired test
:   A comparison of two methods on the same users, through each user's difference. It removes the users' own spread and detects much smaller differences than comparing two separate averages. See [compare results](../../handbook/compare-results.md).

Pareto front
:   The methods that no other method beats on two goals at once, here accuracy and training time. Choose from the front at the budget you can afford.

Patience
:   How many epochs without improvement early stopping waits before it ends training.

Popularity bias
:   The tendency of models to over-recommend popular items. See [popularity bias](popularity-bias.md).

Precision@K
:   The share of the K recommended items that are relevant.

Pre-test
:   recbench's name for all events before the test cutoff (train + validation): everything a model may see.

Promotion
:   Making a lab improvement part of the method: a search-space option (better on some datasets, worse on none) or the new default (better on all five), followed by a re-run of its bake-off job. See [promote a winner](../../handbook/promote.md).

Protocol (evaluation)
:   The full set of evaluation choices. recbench's current one is version 2. See [evaluation protocols](evaluation-protocols.md).

Quick tier
:   The bake-off's data size: about one million events per dataset, sampled by user, with the real test window.

Recall@K
:   The share of a user's relevant items found in the top K (recbench divides by min(K, number of relevant items)).

Registry
:   recbench's plugin mechanism: classes register themselves with a decorator. See
    [registry and catalog](../../codebase/registry-and-catalog.md).

Relative score
:   A method's NDCG@10 divided by the best NDCG@10 on the same dataset. Averaged over datasets, it compares methods on one scale.

Repeat policy
:   Whether items a user already interacted with count as relevant again (`allow_repeats`) or are removed
    (`exclude_seen`).

Retrieval / ranking
:   The two stages of a production recommender: find candidates, then order them. See
    [retrieval and ranking](retrieval-and-ranking.md).

## S–Z

Sampled metrics
:   Ranking the true item against a small random set of negatives. Cheap but unreliable.

Search space
:   The settings a tuning job may try, with their ranges (`configs/tuning/quick.yaml`).

Segment (error analysis)
:   A group of users (by activity, recency, or the popularity of their items) in which a difference is measured separately, to see where a change helps or hurts.

Serendipity
:   Recommendations that are both relevant and unexpected.

Session
:   A burst of activity; recbench starts a new session after 30 minutes without events.

Smoke tier
:   A small user-sampled version of each dataset (about 50,000 events) for quick end-to-end runs.

Softmax
:   Turns a vector of scores into probabilities that sum to 1: $\exp(z_i)/\sum_j\exp(z_j)$.

Sparsity
:   The share of empty cells in the user × item matrix (often above 99.9%).

Split
:   The division of data into training, validation, and test parts.

Step budget
:   The fixed number of optimisation steps an older (held-back) model trains for in recbench (preset-dependent).
    The bake-off's methods train in epochs with early stopping instead.

Sweep
:   Trying one setting at several values while everything else stays fixed, to see how sensitive a method is to it (`python -m recbench.lab sweep`).

Temporal split
:   A split by time: earlier events for training, later ones for testing.

Tier
:   The size of a recbench split: smoke, standard, quick, slice, or full. Adding `-val` (for example `quick-val`)
    gives that tier's validation fold.

Top-K
:   The K highest-scored items for a user (recbench uses K = 10, 20, 50).

Train window
:   Training on only the last N days of events (`train_window_days`); each user keeps their last 10 events, so nobody's profile disappears.

TrainView
:   The object through which recbench models see data: pre-test events only, with no path to test files.

Trial
:   One setting tried by a tuning job: one training run on the validation fold and its score.

Two-stage recommender
:   Cheap models propose candidates (stage 1, retrieval), and a ranker orders them with many features (stage 2). recbench's re-rankers are two-stage methods.

VAE (variational autoencoder)
:   A network that compresses a user's history into a small random code and rebuilds the history from it (MultVAE, RecVAE).

Variant
:   A new setting of an existing method that switches on changed code, with a default that keeps today's behaviour, for example `ease_variant: edlae`.

Warm user / item
:   A user or item that has pre-test history.


Winner's curse
:   Picking the best of many noisy scores always picks a lucky one: its score overstates its quality. The lab chooses on validation and tests once to avoid it.

Workspace
:   A separate place for results: with `workspace: lab`, runs, summaries and reports go to `runs/lab/` and `reports/lab/`, apart from the bake-off's.
