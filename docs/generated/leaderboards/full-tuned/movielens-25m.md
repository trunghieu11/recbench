
**Split at a glance**

|  |  |
|---|---|
| Users / items | 162,541 / 59,047 |
| Events before / after the test cutoff | 22,500,085 / 2,500,010 |
| Test window starts (UTC) | 2018-01-04 02:45:08 |
| Warm eval users (have history and a new test item) | 6,492 |
| Cold test users (no history; popularity is their only option) | 11,950 |
| Share of test interactions that repeat a past item | 0.0% |
| Repeat policies | exclude_seen (primary: exclude_seen) |

### Top-N for warm users (full-catalog ranking)

Sorted by NDCG@10. ≈ marks methods whose 95% confidence interval overlaps the best one's.

| # | Method | NDCG@10 [95% CI] | Recall@10 | HitRate@10 | Coverage@10 | Train s | Score s / 1k users | Peak RSS MB | Personal expl. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ease ≈ | 0.2249 [0.2189, 0.2311] | 0.216 | 0.648 | 0.014 | 75.5 | 0.42 | 8,772 | 1.00 |
| 2 | lgbm_rerank ≈ | 0.2223 [0.2165, 0.2278] | 0.222 | 0.689 | 0.017 | 729.7 | 7.31 | 34,196 | 0.46 |
| 3 | puresvd ≈ | 0.2151 [0.2090, 0.2212] | 0.205 | 0.638 | 0.019 | 38.0 | 0.04 | 3,649 | – |

### Next-item prediction

Can the method put the user's very next item in its top 10?

| # | Method | Next NDCG@10 [95% CI] | Next HitRate@10 |
|---|---|---|---|
| 1 | lgbm_rerank | 0.0592 [0.0553, 0.0637] | 0.121 |
| 2 | ease | 0.0486 [0.0447, 0.0528] | 0.093 |
| 3 | puresvd | 0.0380 [0.0345, 0.0422] | 0.074 |

### Beyond accuracy

| Method | Coverage | Gini ↓ | Popularity pct ↓ | Long-tail share | Novelty | ILD | Serendipity | Calibration KL ↓ | Group gap ↓ | Cold-item recall |
|---|---|---|---|---|---|---|---|---|---|---|
| ease | 0.014 | 0.998 | 0.985 | 0.000 | 10.90 | 0.745 | 0.000 | 1.447 | 0.057 | 0.000 |
| lgbm_rerank | 0.017 | 0.998 | 0.945 | 0.008 | 13.31 | 0.767 | 0.000 | 1.205 | 0.065 | 0.000 |
| puresvd | 0.019 | 0.996 | 0.987 | 0.000 | 10.95 | 0.766 | 0.000 | 1.079 | 0.025 | 0.000 |

### Full ranking vs. sampled (1 + 100 negatives)

If the two rank columns disagree, the cheaper sampled protocol would have changed the conclusion.

| Method | Rank (full) | Rank (sampled) | Sampled NDCG@10 |
|---|---|---|---|
| ease | 1 | 1 | 0.638 |
| lgbm_rerank | 2 | 3 ↓ | 0.411 |
| puresvd | 3 | 2 ↑ | 0.602 |

