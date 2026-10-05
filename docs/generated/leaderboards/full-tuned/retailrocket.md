
**Split at a glance**

|  |  |
|---|---|
| Users / items | 1,407,580 / 235,061 |
| Events before / after the test cutoff | 2,480,490 / 275,611 |
| Test window starts (UTC) | 2015-09-02 17:49:11 |
| Warm eval users (have history and a new test item) | 10,000 |
| Cold test users (no history; popularity is their only option) | 142,102 |
| Share of test interactions that repeat a past item | 2.0% |
| Repeat policies | exclude_seen (primary: exclude_seen) |

### Top-N for warm users (full-catalog ranking)

Sorted by NDCG@10. ≈ marks methods whose 95% confidence interval overlaps the best one's.

| # | Method | NDCG@10 [95% CI] | Recall@10 | HitRate@10 | Coverage@10 | Train s | Score s / 1k users | Peak RSS MB | Personal expl. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | rp3beta ≈ | 0.0275 [0.0250, 0.0302] | 0.041 | 0.057 | 0.159 | 21.1 | 0.35 | 4,899 | 0.89 |
| 2 | ials ≈ | 0.0272 [0.0248, 0.0297] | 0.043 | 0.059 | 0.076 | 2,106 | 0.23 | 8,847 | 1.00 |
| 3 | lgbm_rerank ≈ | 0.0270 [0.0245, 0.0294] | 0.042 | 0.059 | 0.100 | 884.8 | 11.00 | 62,076 | 0.40 |

### Next-item prediction

Can the method put the user's very next item in its top 10?

| # | Method | Next NDCG@10 [95% CI] | Next HitRate@10 |
|---|---|---|---|
| 1 | rp3beta | 0.0253 [0.0227, 0.0278] | 0.042 |
| 2 | ials | 0.0252 [0.0227, 0.0278] | 0.044 |
| 3 | lgbm_rerank | 0.0247 [0.0222, 0.0271] | 0.043 |

### Beyond accuracy

| Method | Coverage | Gini ↓ | Popularity pct ↓ | Long-tail share | Novelty | ILD | Serendipity | Calibration KL ↓ | Group gap ↓ | Cold-item recall |
|---|---|---|---|---|---|---|---|---|---|---|
| rp3beta | 0.159 | 0.927 | 0.824 | 0.329 | 16.57 | 0.606 | 0.000 | 1.854 | 0.006 | 0.000 |
| ials | 0.076 | 0.968 | 0.973 | 0.005 | 14.64 | 0.426 | 0.000 | 1.984 | 0.005 | 0.000 |
| lgbm_rerank | 0.100 | 0.967 | 0.949 | 0.072 | 14.09 | 0.598 | 0.000 | 1.942 | 0.006 | 0.000 |

### Full ranking vs. sampled (1 + 100 negatives)

If the two rank columns disagree, the cheaper sampled protocol would have changed the conclusion.

| Method | Rank (full) | Rank (sampled) | Sampled NDCG@10 |
|---|---|---|---|
| rp3beta | 1 | 3 ↓ | 0.067 |
| ials | 2 | 1 ↑ | 0.282 |
| lgbm_rerank | 3 | 2 ↑ | 0.082 |

