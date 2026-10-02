<!-- generated -->

| Metric | Kind | Direction | Tasks | Meaning |
|---|---|---|---|---|
| `hitrate_at_10` | accuracy | ↑ higher is better | topn | Share of users with at least one relevant item in the top 10. |
| `hitrate_at_20` | accuracy | ↑ higher is better | topn | Share of users with at least one relevant item in the top 20. |
| `hitrate_at_50` | accuracy | ↑ higher is better | topn | Share of users with at least one relevant item in the top 50. |
| `map_at_10` | accuracy | ↑ higher is better | topn | Mean average precision of the top 10. |
| `mrr_at_50` | accuracy | ↑ higher is better | topn | 1 / rank of the first relevant item within the top 50. |
| `ndcg_at_10` | accuracy | ↑ higher is better | topn | Position-aware ranking quality of the top 10 (1 = perfect order). |
| `ndcg_at_20` | accuracy | ↑ higher is better | topn | Position-aware ranking quality of the top 20 (1 = perfect order). |
| `ndcg_at_50` | accuracy | ↑ higher is better | topn | Position-aware ranking quality of the top 50 (1 = perfect order). |
| `next_hitrate_at_10` | accuracy | ↑ higher is better | sequential, session, similar_items, topn | Next-item task: the very next item the user interacted with is in the top 10. |
| `next_ndcg_at_10` | accuracy | ↑ higher is better | sequential, session, similar_items, topn | Next-item task: NDCG@10 with the next item as the only relevant item. |
| `precision_at_10` | accuracy | ↑ higher is better | topn | Share of the 10 recommended slots that are relevant. |
| `recall_at_10` | accuracy | ↑ higher is better | topn | Relevant items found in the top 10 / min(10, #relevant). |
| `recall_at_20` | accuracy | ↑ higher is better | topn | Relevant items found in the top 20 / min(20, #relevant). |
| `recall_at_50` | accuracy | ↑ higher is better | topn | Relevant items found in the top 50 / min(50, #relevant). |
| `calibration_kl_at_10` | beyond | ↓ lower is better | topn | KL divergence between the user's category mix and the list's mix (Steck 2018). |
| `coverage_at_10` | beyond | ↑ higher is better | topn | Share of the catalog that appears in at least one top-10 list. |
| `gini_at_10` | beyond | ↓ lower is better | topn | Inequality of exposure across catalog items in top-10 lists (0 = equal). |
| `ild_at_10` | beyond | ↑ higher is better | topn | Intra-list diversity: mean pairwise category (Jaccard) distance inside a top-10. |
| `long_tail_share_at_10` | beyond | ↑ higher is better | topn | Share of recommended items outside the 20% most popular items. |
| `novelty_at_10` | beyond | ↑ higher is better | topn | Mean self-information -log2 p(item) of recommended items (higher = less obvious). |
| `popularity_percentile_at_10` | beyond | ↓ lower is better | topn | Average popularity percentile of recommended items (1 = most popular). |
| `serendipity_at_10` | beyond | ↑ higher is better | topn | Relevant AND unexpected (not top-5% popular, new category for the user) share of a top-10. |
| `user_group_ndcg_gap_at_10` | beyond | ↓ lower is better | topn | Max - min mean NDCG@10 across light/medium/heavy user groups. |
| `sampled_auc` | diagnostic | ↑ higher is better | ctr, topn | Per-user AUC (GAUC) of the next item vs 100 random unseen items. |
| `sampled_logloss` | diagnostic | ↓ lower is better | ctr | Log loss on the sampled candidates; only for models that output probabilities. |
| `peak_gpu_mb` | efficiency | ↓ lower is better | ctr, sequential, session, similar_items, topn | Peak GPU memory allocated by torch (0 on CPU). |
| `peak_rss_mb` | efficiency | ↓ lower is better | ctr, sequential, session, similar_items, topn | Peak resident memory of the run's process. |
| `score_seconds_per_1k_users` | efficiency | ↓ lower is better | ctr, sequential, session, similar_items, topn | Batch scoring time for 1,000 users against the whole catalog. |
| `train_seconds` | efficiency | ↓ lower is better | ctr, sequential, session, similar_items, topn | Wall time of fit() on the active hardware. |
| `personal_explanation_rate` | explainability | ↑ higher is better | ctr, sequential, session, similar_items, topn | Share of sampled recommendations whose explanation cites one of the user's own history items. |
| `sampled_hitrate_at_10` | sampled | ↑ higher is better | sequential, session, similar_items, topn | Next item ranked in the top 10 among itself + 100 random unseen items. |
| `sampled_ndcg_at_10` | sampled | ↑ higher is better | sequential, session, similar_items, topn | NDCG@10 of the next item among itself + 100 random unseen items. |
| `item_cold_recall_at_10` | slice | ↑ higher is better | topn | Recall@10 counting only relevant items with no pre-test history. |
