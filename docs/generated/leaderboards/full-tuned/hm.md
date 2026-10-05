
**Split at a glance**

|  |  |
|---|---|
| Users / items | 1,362,281 / 104,547 |
| Events before / after the test cutoff | 31,548,013 / 240,311 |
| Test window starts (UTC) | 2020-09-16 00:00:00 |
| Warm eval users (have history and a new test item) | 10,000 |
| Cold test users (no history; popularity is their only option) | 5,572 |
| Share of test interactions that repeat a past item | 3.9% |
| Repeat policies | exclude_seen (primary: exclude_seen) |

### Top-N for warm users (full-catalog ranking)

Sorted by NDCG@10. ≈ marks methods whose 95% confidence interval overlaps the best one's.

| # | Method | NDCG@10 [95% CI] | Recall@10 | HitRate@10 | Coverage@10 | Train s | Score s / 1k users | Peak RSS MB | Personal expl. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | lgbm_rerank ≈ | 0.0283 [0.0260, 0.0305] | 0.042 | 0.092 | 0.034 | 823.6 | 5.98 | 34,913 | 0.44 |
| 2 | ease | 0.0213 [0.0192, 0.0234] | 0.031 | 0.069 | 0.064 | 24.9 | 0.37 | 6,884 | 0.98 |
| 3 | itemknn | 0.0169 [0.0151, 0.0187] | 0.025 | 0.054 | 0.069 | 55.7 | 0.33 | 6,234 | 1.00 |

### Next-item prediction

Can the method put the user's very next item in its top 10?

| # | Method | Next NDCG@10 [95% CI] | Next HitRate@10 |
|---|---|---|---|
| 1 | lgbm_rerank | 0.0217 [0.0195, 0.0241] | 0.040 |
| 2 | ease | 0.0170 [0.0150, 0.0191] | 0.031 |
| 3 | itemknn | 0.0137 [0.0118, 0.0157] | 0.025 |

### Beyond accuracy

| Method | Coverage | Gini ↓ | Popularity pct ↓ | Long-tail share | Novelty | ILD | Serendipity | Calibration KL ↓ | Group gap ↓ | Cold-item recall |
|---|---|---|---|---|---|---|---|---|---|---|
| lgbm_rerank | 0.034 | 0.995 | 0.916 | 0.146 | 14.02 | 0.708 | 0.000 | 3.457 | 0.006 | 0.000 |
| ease | 0.064 | 0.988 | 0.930 | 0.094 | 13.86 | 0.680 | 0.000 | 3.874 | 0.004 | 0.000 |
| itemknn | 0.069 | 0.985 | 0.957 | 0.048 | 13.29 | 0.654 | 0.000 | 3.645 | 0.011 | 0.000 |

### Full ranking vs. sampled (1 + 100 negatives)

If the two rank columns disagree, the cheaper sampled protocol would have changed the conclusion.

| Method | Rank (full) | Rank (sampled) | Sampled NDCG@10 |
|---|---|---|---|
| lgbm_rerank | 1 | 3 ↓ | 0.182 |
| ease | 2 | 1 ↑ | 0.534 |
| itemknn | 3 | 2 ↑ | 0.280 |

