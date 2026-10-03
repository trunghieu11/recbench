
**Split at a glance**

|  |  |
|---|---|
| Users / items | 992 / 176,892 |
| Events before / after the test cutoff | 17,235,780 / 1,915,088 |
| Test window starts (UTC) | 2009-02-02 16:02:09 |
| Warm eval users (have history and a new test item) | 851 |
| Cold test users (no history; popularity is their only option) | 33 |
| Share of test interactions that repeat a past item | 61.7% |
| Repeat policies | exclude_seen, allow_repeats (primary: exclude_seen) |

### Top-N for warm users (full-catalog ranking)

Sorted by NDCG@10. ≈ marks methods whose 95% confidence interval overlaps the best one's.

| # | Method | NDCG@10 [95% CI] | Recall@10 | HitRate@10 | Coverage@10 | Train s | Score s / 1k users | Peak RSS MB | Personal expl. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ease ≈ | 0.0945 [0.0830, 0.1043] | 0.093 | 0.423 | 0.008 | 127.1 | 5.73 | 31,905 | 1.00 |
| 2 | bpr_mf ≈ | 0.0731 [0.0634, 0.0832] | 0.069 | 0.345 | 0.024 | 32.9 | 0.14 | 3,969 | 1.00 |
| 3 | itemknn | 0.0729 [0.0625, 0.0819] | 0.073 | 0.329 | 0.003 | 52.4 | 1.08 | 4,345 | 1.00 |
| 4 | sasrec | 0.0615 [0.0538, 0.0698] | 0.060 | 0.325 | 0.018 | 2,084 | 0.45 | 4,869 | 1.00 |
| 5 | dcnv2 | 0.0529 [0.0456, 0.0608] | 0.051 | 0.282 | 0.008 | 908.9 | 17.14 | 4,739 | 1.00 |
| 6 | ials | 0.0485 [0.0419, 0.0559] | 0.049 | 0.273 | 0.016 | 21.5 | 0.10 | 3,970 | 1.00 |
| 7 | most_popular | 0.0478 [0.0409, 0.0547] | 0.051 | 0.277 | 0.001 | 0.2 | 0.09 | 3,951 | 0.00 |
| 8 | text_hash_tower | 0.0020 [0.0010, 0.0032] | 0.002 | 0.019 | 0.030 | 669.4 | 0.50 | 4,799 | 1.00 |
| 9 | random | 0.0006 [0.0002, 0.0012] | 0.001 | 0.006 | 0.047 | 0.0 | 0.61 | 3,849 | 0.00 |

### Next-item prediction

Can the method put the user's very next item in its top 10?

| # | Method | Next NDCG@10 [95% CI] | Next HitRate@10 |
|---|---|---|---|
| 1 | ease | 0.0118 [0.0069, 0.0170] | 0.026 |
| 2 | sasrec | 0.0085 [0.0046, 0.0128] | 0.020 |
| 3 | dcnv2 | 0.0075 [0.0029, 0.0129] | 0.012 |
| 4 | itemknn | 0.0072 [0.0036, 0.0112] | 0.015 |
| 5 | most_popular | 0.0057 [0.0021, 0.0098] | 0.011 |
| 6 | bpr_mf | 0.0054 [0.0022, 0.0093] | 0.013 |
| 7 | ials | 0.0038 [0.0016, 0.0065] | 0.011 |
| 8 | text_hash_tower | 0.0004 [0.0000, 0.0012] | 0.001 |
| 9 | random | 0.0000 [0.0000, 0.0000] | 0.000 |

### Same users, repeats allowed

Re-listening/re-buying counts as a hit here.

| Method | NDCG@10 (new items only) | NDCG@10 (repeats allowed) |
|---|---|---|
| ease | 0.094 | 0.476 |
| bpr_mf | 0.073 | 0.418 |
| itemknn | 0.073 | 0.388 |
| sasrec | 0.061 | 0.486 |
| dcnv2 | 0.053 | 0.313 |
| ials | 0.049 | 0.106 |
| most_popular | 0.048 | 0.184 |
| text_hash_tower | 0.002 | 0.102 |
| random | 0.001 | 0.001 |

### Beyond accuracy

| Method | Coverage | Gini ↓ | Popularity pct ↓ | Long-tail share | Novelty | ILD | Serendipity | Calibration KL ↓ | Group gap ↓ | Cold-item recall |
|---|---|---|---|---|---|---|---|---|---|---|
| ease | 0.008 | 0.997 | 0.997 | 0.002 | 10.54 | – | 0.000 | – | 0.048 | 0.000 |
| bpr_mf | 0.024 | 0.985 | 0.973 | 0.010 | 13.69 | – | 0.006 | – | 0.064 | 0.000 |
| itemknn | 0.003 | 0.999 | 0.998 | 0.001 | 9.49 | – | 0.000 | – | 0.036 | 0.000 |
| sasrec | 0.018 | 0.991 | 0.981 | 0.018 | 12.39 | – | 0.002 | – | 0.024 | 0.000 |
| dcnv2 | 0.008 | 0.998 | 0.997 | 0.002 | 10.38 | – | 0.000 | – | 0.026 | 0.000 |
| ials | 0.016 | 0.991 | 0.990 | 0.000 | 12.50 | – | 0.000 | – | 0.042 | 0.000 |
| most_popular | 0.001 | 1.000 | 0.999 | 0.000 | 9.72 | – | 0.000 | – | 0.021 | 0.000 |
| text_hash_tower | 0.030 | 0.980 | 0.578 | 0.758 | 20.90 | – | 0.001 | – | 0.000 | 0.002 |
| random | 0.047 | 0.954 | 0.542 | 0.801 | 21.24 | – | 0.000 | – | 0.001 | 0.000 |

### Full ranking vs. sampled (1 + 100 negatives)

If the two rank columns disagree, the cheaper sampled protocol would have changed the conclusion.

| Method | Rank (full) | Rank (sampled) | Sampled NDCG@10 |
|---|---|---|---|
| ease | 1 | 1 | 0.491 |
| bpr_mf | 2 | 7 ↓ | 0.377 |
| itemknn | 3 | 4 ↓ | 0.435 |
| sasrec | 4 | 3 ↑ | 0.460 |
| dcnv2 | 5 | 6 ↓ | 0.398 |
| ials | 6 | 2 ↑ | 0.465 |
| most_popular | 7 | 5 ↑ | 0.431 |
| text_hash_tower | 8 | 8 | 0.127 |
| random | 9 | 9 | 0.039 |

### Experimental (not ranked)

| Method | NDCG@10 | Next NDCG@10 |
|---|---|---|
| tiger_lite | 0.0547 [0.0470, 0.0627] | 0.007 |

### Did not run

| Method | Status | Reason |
|---|---|---|
| multimodal_tower | unsupported | no item images on disk |
| bert4rec | timeout | exceeded 120.0 minutes (wall clock) |

