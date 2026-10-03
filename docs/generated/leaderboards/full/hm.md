
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
| 1 | ease ≈ | 0.0098 [0.0085, 0.0110] | 0.015 | 0.032 | 0.082 | 143.3 | 0.56 | 32,869 | 1.00 |
| 2 | most_popular ≈ | 0.0097 [0.0085, 0.0108] | 0.015 | 0.042 | 0.000 | 0.3 | 0.06 | 6,213 | 0.00 |
| 3 | itemknn ≈ | 0.0083 [0.0072, 0.0094] | 0.013 | 0.030 | 0.156 | 111.6 | 0.33 | 6,426 | 1.00 |
| 4 | ials ≈ | 0.0081 [0.0070, 0.0092] | 0.012 | 0.029 | 0.031 | 231.9 | 0.06 | 6,850 | 1.00 |
| 5 | sasrec | 0.0042 [0.0035, 0.0050] | 0.006 | 0.018 | 0.038 | 1,782 | 0.30 | 7,133 | 1.00 |
| 6 | dcnv2 | 0.0028 [0.0023, 0.0034] | 0.004 | 0.015 | 0.006 | 794.0 | 10.10 | 6,985 | 1.00 |
| 7 | text_hash_tower | 0.0028 [0.0022, 0.0035] | 0.005 | 0.011 | 0.088 | 1,639 | 0.28 | 7,043 | 1.00 |
| 8 | bpr_mf | 0.0023 [0.0018, 0.0028] | 0.004 | 0.010 | 0.337 | 990.2 | 0.07 | 6,870 | 1.00 |
| 9 | random | 0.0001 [0.0000, 0.0002] | 0.000 | 0.000 | 0.616 | 0.0 | 0.38 | 6,204 | 0.00 |

### Next-item prediction

Can the method put the user's very next item in its top 10?

| # | Method | Next NDCG@10 [95% CI] | Next HitRate@10 |
|---|---|---|---|
| 1 | ease | 0.0081 [0.0068, 0.0095] | 0.015 |
| 2 | itemknn | 0.0068 [0.0055, 0.0081] | 0.014 |
| 3 | ials | 0.0066 [0.0054, 0.0079] | 0.013 |
| 4 | most_popular | 0.0060 [0.0049, 0.0071] | 0.013 |
| 5 | sasrec | 0.0035 [0.0027, 0.0045] | 0.007 |
| 6 | dcnv2 | 0.0022 [0.0015, 0.0029] | 0.005 |
| 7 | text_hash_tower | 0.0020 [0.0014, 0.0027] | 0.004 |
| 8 | bpr_mf | 0.0015 [0.0010, 0.0021] | 0.003 |
| 9 | random | 0.0001 [0.0000, 0.0002] | 0.000 |

### Beyond accuracy

| Method | Coverage | Gini ↓ | Popularity pct ↓ | Long-tail share | Novelty | ILD | Serendipity | Calibration KL ↓ | Group gap ↓ | Cold-item recall |
|---|---|---|---|---|---|---|---|---|---|---|
| ease | 0.082 | 0.976 | 0.967 | 0.026 | 13.14 | 0.758 | 0.000 | 2.813 | 0.009 | 0.000 |
| most_popular | 0.000 | 1.000 | 0.988 | 0.000 | 12.66 | 0.848 | 0.000 | 4.275 | 0.001 | 0.000 |
| itemknn | 0.156 | 0.959 | 0.935 | 0.092 | 13.28 | 0.705 | 0.000 | 3.029 | 0.008 | 0.000 |
| ials | 0.031 | 0.991 | 0.994 | 0.000 | 12.34 | 0.662 | 0.000 | 3.321 | 0.003 | 0.000 |
| sasrec | 0.038 | 0.994 | 0.972 | 0.012 | 13.31 | 0.796 | 0.000 | 3.853 | 0.002 | 0.000 |
| dcnv2 | 0.006 | 0.999 | 0.998 | 0.000 | 10.93 | 0.733 | 0.000 | 4.501 | 0.002 | 0.000 |
| text_hash_tower | 0.088 | 0.985 | 0.876 | 0.200 | 14.70 | 0.462 | 0.000 | 4.197 | 0.004 | 0.000 |
| bpr_mf | 0.337 | 0.804 | 0.775 | 0.519 | 16.32 | 0.735 | 0.000 | 3.492 | 0.003 | 0.000 |
| random | 0.616 | 0.534 | 0.505 | 0.799 | 18.93 | 0.953 | 0.000 | 4.302 | 0.000 | 0.000 |

### Full ranking vs. sampled (1 + 100 negatives)

If the two rank columns disagree, the cheaper sampled protocol would have changed the conclusion.

| Method | Rank (full) | Rank (sampled) | Sampled NDCG@10 |
|---|---|---|---|
| ease | 1 | 2 ↓ | 0.324 |
| most_popular | 2 | 1 ↑ | 0.585 |
| itemknn | 3 | 8 ↓ | 0.163 |
| ials | 4 | 5 ↓ | 0.278 |
| sasrec | 5 | 3 ↑ | 0.317 |
| dcnv2 | 6 | 7 ↓ | 0.202 |
| text_hash_tower | 7 | 6 ↑ | 0.243 |
| bpr_mf | 8 | 4 ↑ | 0.280 |
| random | 9 | 9 | 0.044 |

### Experimental (not ranked)

| Method | NDCG@10 | Next NDCG@10 |
|---|---|---|
| tiger_lite | 0.0073 [0.0062, 0.0084] | 0.006 |

### Did not run

| Method | Status | Reason |
|---|---|---|
| multimodal_tower | unsupported | no item images on disk |

