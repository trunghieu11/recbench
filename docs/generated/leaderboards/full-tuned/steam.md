
**Split at a glance**

|  |  |
|---|---|
| Users / items | 2,567,538 / 15,474 |
| Events before / after the test cutoff | 7,011,186 / 781,883 |
| Test window starts (UTC) | 2017-10-17 00:00:00 |
| Warm eval users (have history and a new test item) | 10,000 |
| Cold test users (no history; popularity is their only option) | 234,775 |
| Share of test interactions that repeat a past item | 1.3% |
| Repeat policies | exclude_seen (primary: exclude_seen) |

### Top-N for warm users (full-catalog ranking)

Sorted by NDCG@10. ≈ marks methods whose 95% confidence interval overlaps the best one's.

| # | Method | NDCG@10 [95% CI] | Recall@10 | HitRate@10 | Coverage@10 | Train s | Score s / 1k users | Peak RSS MB | Personal expl. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ease ≈ | 0.0582 [0.0549, 0.0614] | 0.103 | 0.145 | 0.135 | 20.1 | 0.47 | 3,700 | 1.00 |
| 2 | sansa ≈ | 0.0568 [0.0536, 0.0599] | 0.105 | 0.147 | 0.148 | 366.7 | 1.21 | 3,735 | – |
| 3 | lgbm_rerank ≈ | 0.0550 [0.0520, 0.0582] | 0.101 | 0.140 | 0.074 | 341.1 | 3.04 | 8,822 | 0.82 |

### Next-item prediction

Can the method put the user's very next item in its top 10?

| # | Method | Next NDCG@10 [95% CI] | Next HitRate@10 |
|---|---|---|---|
| 1 | ease | 0.0555 [0.0522, 0.0590] | 0.106 |
| 2 | sansa | 0.0543 [0.0511, 0.0579] | 0.108 |
| 3 | lgbm_rerank | 0.0535 [0.0502, 0.0569] | 0.105 |

### Beyond accuracy

| Method | Coverage | Gini ↓ | Popularity pct ↓ | Long-tail share | Novelty | ILD | Serendipity | Calibration KL ↓ | Group gap ↓ | Cold-item recall |
|---|---|---|---|---|---|---|---|---|---|---|
| ease | 0.135 | 0.993 | 0.991 | 0.008 | 7.85 | 0.716 | 0.000 | 1.147 | 0.011 | 0.000 |
| sansa | 0.148 | 0.993 | 0.985 | 0.020 | 8.16 | 0.731 | 0.000 | 1.284 | 0.007 | 0.000 |
| lgbm_rerank | 0.074 | 0.996 | 0.980 | 0.004 | 8.99 | 0.715 | 0.000 | 1.214 | 0.011 | 0.000 |

### Full ranking vs. sampled (1 + 100 negatives)

If the two rank columns disagree, the cheaper sampled protocol would have changed the conclusion.

| Method | Rank (full) | Rank (sampled) | Sampled NDCG@10 |
|---|---|---|---|
| ease | 1 | 1 | 0.521 |
| sansa | 2 | 2 | 0.479 |
| lgbm_rerank | 3 | 3 | 0.329 |

