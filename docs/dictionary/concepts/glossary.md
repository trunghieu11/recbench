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

Catalog
:   The set of all items that can be recommended.

Cold start
:   Making recommendations for users or items without history. See [cold start](cold-start.md).

Collaborative filtering
:   Learning from the behaviour of many users ("people like you liked..."). See
    [collaborative, content-based, hybrid](collaborative-content-hybrid.md).

Confidence interval (CI)
:   A range that likely contains the true value of a metric. recbench reports 95% bootstrap intervals.

Content-based
:   Using item descriptions (text, categories, images) to recommend.

Cosine similarity
:   The dot product of two vectors divided by their lengths; it compares directions, range [−1, 1].

Coverage
:   The share of the catalog that a method ever recommends. See [coverage and popularity](../metrics/coverage-and-popularity.md).

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

Dot product
:   The sum of element-wise products of two vectors; the usual score between a user and an item vector.

Embedding
:   A learned vector representing a user, an item, or anything else. See [embeddings](embeddings.md).

Epoch
:   One full pass over the training data. recbench trains for a fixed number of *steps* instead.

Explainability
:   The ability to say why an item was recommended. See [explainability](explainability.md).

Explicit feedback
:   Users stating their opinion, such as star ratings. See [feedback types](feedback-types.md).

Full ranking
:   Ranking the true item against the entire catalog (recbench's main protocol). See
    [full ranking vs sampled](../metrics/sampled-vs-full.md).

Gini coefficient
:   A measure of inequality; 0 = everything shown equally often, 1 = one item gets all the exposure.

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

Negative sampling
:   Picking non-interacted items as stand-in negatives. See [negative sampling](negative-sampling.md).

Next-item prediction
:   Predicting the very next item a user will interact with. See [sequential models](sequential-and-session.md).

Novelty
:   How unexpected (unpopular) recommended items are. See [novelty, diversity, serendipity](../metrics/novelty-diversity-serendipity.md).

## P–R

Padding
:   Filler (item 0) used to make histories of different lengths fit into one rectangular batch.

Popularity bias
:   The tendency of models to over-recommend popular items. See [popularity bias](popularity-bias.md).

Precision@K
:   The share of the K recommended items that are relevant.

Pre-test
:   recbench's name for all events before the test cutoff (train + validation): everything a model may see.

Protocol (evaluation)
:   The full set of evaluation choices. recbench's current one is version 2. See [evaluation protocols](evaluation-protocols.md).

Recall@K
:   The share of a user's relevant items found in the top K (recbench divides by min(K, number of relevant items)).

Registry
:   recbench's plugin mechanism: classes register themselves with a decorator. See
    [registry and catalog](../../codebase/registry-and-catalog.md).

Repeat policy
:   Whether items a user already interacted with count as relevant again (`allow_repeats`) or are removed
    (`exclude_seen`).

Retrieval / ranking
:   The two stages of a production recommender: find candidates, then order them. See
    [retrieval and ranking](retrieval-and-ranking.md).

## S–Z

Sampled metrics
:   Ranking the true item against a small random set of negatives. Cheap but unreliable.

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
:   The fixed number of optimisation steps a model trains for in recbench (preset-dependent).

Temporal split
:   A split by time: earlier events for training, later ones for testing.

Tier
:   The size of a recbench split: smoke, standard, slice, or full.

Top-K
:   The K highest-scored items for a user (recbench uses K = 10, 20, 50).

TrainView
:   The object through which recbench models see data: pre-test events only, with no path to test files.

Warm user / item
:   A user or item that has pre-test history.
