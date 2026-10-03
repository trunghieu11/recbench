
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
| 1 | ease ≈ | 0.0242 [0.0219, 0.0266] | 0.036 | 0.051 | 0.071 | 123.0 | 0.39 | 30,954 | 0.77 |
| 2 | sasrec ≈ | 0.0221 [0.0200, 0.0244] | 0.032 | 0.045 | 0.155 | 1,838 | 0.66 | 6,985 | 1.00 |
| 3 | itemknn ≈ | 0.0213 [0.0191, 0.0237] | 0.032 | 0.045 | 0.155 | 10.7 | 0.23 | 5,619 | 0.91 |
| 4 | lightgcn | 0.0195 [0.0176, 0.0216] | 0.031 | 0.044 | 0.071 | 5,515 | 0.21 | 7,835 | 1.00 |
| 5 | ials | 0.0180 [0.0159, 0.0199] | 0.029 | 0.041 | 0.018 | 204.2 | 0.12 | 6,838 | 1.00 |
| 6 | bpr_mf | 0.0054 [0.0043, 0.0065] | 0.008 | 0.014 | 0.015 | 79.0 | 0.15 | 6,825 | 1.00 |
| 7 | most_popular | 0.0034 [0.0026, 0.0042] | 0.007 | 0.011 | 0.000 | 0.0 | 0.14 | 6,114 | 0.00 |
| 8 | text_hash_tower | 0.0024 [0.0017, 0.0030] | 0.004 | 0.007 | 0.236 | 1,676 | 0.65 | 6,928 | 1.00 |
| 9 | dcnv2 | 0.0013 [0.0008, 0.0017] | 0.002 | 0.004 | 0.099 | 838.3 | 23.14 | 6,824 | 1.00 |
| 10 | random | 0.0000 [0.0000, 0.0000] | 0.000 | 0.000 | 0.346 | 0.0 | 0.82 | 6,105 | 0.00 |

### Next-item prediction

Can the method put the user's very next item in its top 10?

| # | Method | Next NDCG@10 [95% CI] | Next HitRate@10 |
|---|---|---|---|
| 1 | ease | 0.0221 [0.0198, 0.0244] | 0.037 |
| 2 | sasrec | 0.0207 [0.0183, 0.0231] | 0.033 |
| 3 | itemknn | 0.0195 [0.0172, 0.0219] | 0.032 |
| 4 | lightgcn | 0.0181 [0.0161, 0.0202] | 0.032 |
| 5 | ials | 0.0167 [0.0146, 0.0187] | 0.030 |
| 6 | bpr_mf | 0.0049 [0.0038, 0.0060] | 0.009 |
| 7 | most_popular | 0.0029 [0.0021, 0.0037] | 0.006 |
| 8 | text_hash_tower | 0.0023 [0.0016, 0.0029] | 0.005 |
| 9 | dcnv2 | 0.0010 [0.0006, 0.0014] | 0.002 |
| 10 | random | 0.0000 [0.0000, 0.0000] | 0.000 |

### Beyond accuracy

| Method | Coverage | Gini ↓ | Popularity pct ↓ | Long-tail share | Novelty | ILD | Serendipity | Calibration KL ↓ | Group gap ↓ | Cold-item recall |
|---|---|---|---|---|---|---|---|---|---|---|
| ease | 0.071 | 0.977 | 0.952 | 0.012 | 15.07 | 0.680 | 0.000 | 3.063 | 0.008 | 0.000 |
| sasrec | 0.155 | 0.923 | 0.905 | 0.155 | 15.64 | 0.691 | 0.000 | 2.771 | 0.000 | 0.000 |
| itemknn | 0.155 | 0.929 | 0.731 | 0.514 | 17.66 | 0.732 | 0.000 | 2.533 | 0.003 | 0.000 |
| lightgcn | 0.071 | 0.979 | 0.966 | 0.047 | 14.17 | 0.606 | 0.000 | 2.693 | 0.002 | 0.000 |
| ials | 0.018 | 0.994 | 0.992 | 0.000 | 13.41 | 0.560 | 0.000 | 3.090 | 0.003 | 0.000 |
| bpr_mf | 0.015 | 0.998 | 0.907 | 0.206 | 15.99 | 0.907 | 0.000 | 5.407 | 0.005 | 0.000 |
| most_popular | 0.000 | 1.000 | 1.000 | 0.000 | 11.16 | 0.934 | 0.000 | 6.277 | 0.000 | 0.000 |
| text_hash_tower | 0.236 | 0.852 | 0.655 | 0.662 | 18.40 | 0.125 | 0.000 | 1.217 | 0.000 | 0.003 |
| dcnv2 | 0.099 | 0.967 | 0.956 | 0.061 | 14.24 | 0.904 | 0.000 | 5.110 | 0.001 | 0.000 |
| random | 0.346 | 0.707 | 0.565 | 0.800 | 19.09 | 0.995 | 0.000 | 6.501 | 0.000 | 0.000 |

### Full ranking vs. sampled (1 + 100 negatives)

If the two rank columns disagree, the cheaper sampled protocol would have changed the conclusion.

| Method | Rank (full) | Rank (sampled) | Sampled NDCG@10 |
|---|---|---|---|
| ease | 1 | 5 ↓ | 0.238 |
| sasrec | 2 | 6 ↓ | 0.236 |
| itemknn | 3 | 9 ↓ | 0.069 |
| lightgcn | 4 | 3 ↑ | 0.285 |
| ials | 5 | 4 ↑ | 0.278 |
| bpr_mf | 6 | 8 ↓ | 0.145 |
| most_popular | 7 | 1 ↑ | 0.322 |
| text_hash_tower | 8 | 2 ↑ | 0.303 |
| dcnv2 | 9 | 7 ↑ | 0.214 |
| random | 10 | 10 | 0.046 |

### Experimental (not ranked)

| Method | NDCG@10 | Next NDCG@10 |
|---|---|---|
| tiger_lite | 0.0148 [0.0131, 0.0167] | 0.013 |

### Did not run

| Method | Status | Reason |
|---|---|---|
| multimodal_tower | unsupported | no item images on disk |

