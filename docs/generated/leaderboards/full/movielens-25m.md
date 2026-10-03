
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
| 1 | ease ≈ | 0.1950 [0.1887, 0.2008] | 0.184 | 0.599 | 0.044 | 178.8 | 3.68 | 32,385 | 1.00 |
| 2 | most_popular ≈ | 0.1917 [0.1861, 0.1974] | 0.190 | 0.616 | 0.004 | 0.2 | 0.03 | 3,948 | 0.00 |
| 3 | ials | 0.1504 [0.1452, 0.1553] | 0.147 | 0.532 | 0.054 | 118.1 | 0.04 | 3,986 | 1.00 |
| 4 | lightgcn | 0.1457 [0.1403, 0.1509] | 0.140 | 0.490 | 0.015 | 6,956 | 0.06 | 5,749 | 1.00 |
| 5 | itemknn | 0.1421 [0.1366, 0.1473] | 0.135 | 0.474 | 0.012 | 87.2 | 0.37 | 3,966 | 1.00 |
| 6 | dcnv2 | 0.1362 [0.1311, 0.1413] | 0.131 | 0.472 | 0.024 | 626.3 | 5.73 | 4,666 | 1.00 |
| 7 | sasrec | 0.1261 [0.1211, 0.1307] | 0.123 | 0.486 | 0.024 | 1,816 | 0.19 | 4,817 | 1.00 |
| 8 | bpr_mf | 0.1244 [0.1188, 0.1293] | 0.118 | 0.457 | 0.140 | 804.0 | 0.04 | 3,985 | 1.00 |
| 9 | text_hash_tower | 0.0812 [0.0778, 0.0848] | 0.078 | 0.364 | 0.036 | 1,265 | 0.17 | 4,729 | 1.00 |
| 10 | random | 0.0017 [0.0013, 0.0021] | 0.002 | 0.015 | 0.667 | 0.0 | 0.22 | 3,947 | 0.00 |

### Next-item prediction

Can the method put the user's very next item in its top 10?

| # | Method | Next NDCG@10 [95% CI] | Next HitRate@10 |
|---|---|---|---|
| 1 | most_popular | 0.0427 [0.0393, 0.0464] | 0.090 |
| 2 | ease | 0.0284 [0.0253, 0.0319] | 0.053 |
| 3 | lightgcn | 0.0226 [0.0199, 0.0256] | 0.044 |
| 4 | sasrec | 0.0189 [0.0165, 0.0217] | 0.039 |
| 5 | ials | 0.0186 [0.0162, 0.0215] | 0.037 |
| 6 | dcnv2 | 0.0180 [0.0155, 0.0204] | 0.036 |
| 7 | itemknn | 0.0174 [0.0149, 0.0200] | 0.034 |
| 8 | bpr_mf | 0.0144 [0.0123, 0.0168] | 0.029 |
| 9 | text_hash_tower | 0.0072 [0.0058, 0.0087] | 0.016 |
| 10 | random | 0.0000 [0.0000, 0.0001] | 0.000 |

### Beyond accuracy

| Method | Coverage | Gini ↓ | Popularity pct ↓ | Long-tail share | Novelty | ILD | Serendipity | Calibration KL ↓ | Group gap ↓ | Cold-item recall |
|---|---|---|---|---|---|---|---|---|---|---|
| ease | 0.044 | 0.990 | 0.992 | 0.000 | 10.53 | 0.786 | 0.000 | 0.804 | 0.023 | 0.000 |
| most_popular | 0.004 | 1.000 | 0.977 | 0.000 | 11.32 | 0.728 | 0.000 | 1.570 | 0.084 | 0.000 |
| ials | 0.054 | 0.985 | 0.989 | 0.000 | 10.88 | 0.771 | 0.000 | 0.913 | 0.054 | 0.000 |
| lightgcn | 0.015 | 0.998 | 0.997 | 0.001 | 9.65 | 0.780 | 0.000 | 1.321 | 0.067 | 0.000 |
| itemknn | 0.012 | 0.998 | 0.997 | 0.001 | 9.76 | 0.752 | 0.000 | 1.259 | 0.060 | 0.000 |
| dcnv2 | 0.024 | 0.997 | 0.995 | 0.000 | 9.96 | 0.767 | 0.000 | 1.206 | 0.046 | 0.000 |
| sasrec | 0.024 | 0.996 | 0.983 | 0.000 | 11.44 | 0.757 | 0.000 | 1.357 | 0.048 | 0.000 |
| bpr_mf | 0.140 | 0.955 | 0.935 | 0.052 | 13.76 | 0.696 | 0.000 | 1.375 | 0.071 | 0.000 |
| text_hash_tower | 0.036 | 0.995 | 0.928 | 0.107 | 12.84 | 0.668 | 0.000 | 1.069 | 0.061 | 0.000 |
| random | 0.667 | 0.504 | 0.541 | 0.807 | 21.35 | 0.840 | 0.000 | 2.049 | 0.001 | 0.001 |

### Full ranking vs. sampled (1 + 100 negatives)

If the two rank columns disagree, the cheaper sampled protocol would have changed the conclusion.

| Method | Rank (full) | Rank (sampled) | Sampled NDCG@10 |
|---|---|---|---|
| ease | 1 | 4 ↓ | 0.535 |
| most_popular | 2 | 1 ↑ | 0.639 |
| ials | 3 | 2 ↑ | 0.552 |
| lightgcn | 4 | 5 ↓ | 0.511 |
| itemknn | 5 | 7 ↓ | 0.488 |
| dcnv2 | 6 | 6 | 0.509 |
| sasrec | 7 | 3 ↑ | 0.542 |
| bpr_mf | 8 | 8 | 0.433 |
| text_hash_tower | 9 | 9 | 0.264 |
| random | 10 | 10 | 0.043 |

### Experimental (not ranked)

| Method | NDCG@10 | Next NDCG@10 |
|---|---|---|
| tiger_lite | 0.1029 [0.0986, 0.1068] | 0.012 |

### Did not run

| Method | Status | Reason |
|---|---|---|
| xsimgcl | failed | ModuleNotFoundError: No module named 'numba' |
| bert4rec | failed | ModuleNotFoundError: No module named 'kmeans_pytorch' |
| s3rec | failed | ModuleNotFoundError: No module named 'kmeans_pytorch' |
| hstu | timeout | exceeded 480.0 minutes (wall clock) |
| din | failed | OutOfMemoryError: CUDA out of memory. Tried to allocate 31.41 GiB. GPU 0 has a total capacity of 47.37 GiB of which 11.10 GiB is free. Process 1235701 has 36.26 |
| multimodal_tower | unsupported | no item images on disk |

