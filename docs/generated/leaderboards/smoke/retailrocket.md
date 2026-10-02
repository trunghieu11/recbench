
**Split at a glance**

|  |  |
|---|---|
| Users / items | 10,365 / 23,773 |
| Events before / after the test cutoff | 37,105 / 12,859 |
| Test window starts (UTC) | 2015-09-02 17:49:11 |
| Warm eval users (have history and a new test item) | 4,230 |
| Cold test users (no history; popularity is their only option) | 561 |
| Share of test interactions that repeat a past item | 12.5% |
| Repeat policies | exclude_seen (primary: exclude_seen) |

### Top-N for warm users (full-catalog ranking)

Sorted by NDCG@10. ≈ marks methods whose 95% confidence interval overlaps the best one's.

| # | Method | NDCG@10 [95% CI] | Recall@10 | HitRate@10 | Coverage@10 | Train s | Score s / 1k users | Peak RSS MB | Personal expl. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ease ≈ | 0.0062 [0.0035, 0.0091] | 0.008 | 0.014 | 0.169 | 40.2 | 0.09 | 7,684 | 0.53 |
| 2 | itemknn ≈ | 0.0049 [0.0024, 0.0079] | 0.006 | 0.009 | 0.183 | 0.1 | 0.00 | 1,416 | 0.53 |
| 3 | lightgcn ≈ | 0.0045 [0.0024, 0.0070] | 0.007 | 0.010 | 0.130 | 3.8 | 0.01 | 1,557 | 1.00 |
| 4 | ials ≈ | 0.0040 [0.0020, 0.0065] | 0.005 | 0.011 | 0.067 | 0.3 | 0.01 | 1,417 | 1.00 |
| 5 | xsimgcl ≈ | 0.0028 [0.0014, 0.0046] | 0.005 | 0.008 | 0.090 | 7.3 | 0.01 | 1,859 | 1.00 |
| 6 | bert4rec | 0.0016 [0.0001, 0.0032] | 0.002 | 0.002 | 0.010 | 27.3 | 0.19 | 2,507 | 1.00 |
| 7 | dcnv2 | 0.0014 [0.0002, 0.0030] | 0.002 | 0.003 | 0.123 | 4.4 | 3.49 | 6,219 | 1.00 |
| 8 | most_popular | 0.0011 [0.0003, 0.0021] | 0.002 | 0.004 | 0.001 | 0.0 | 0.00 | 1,556 | 0.00 |
| 9 | s3rec | 0.0010 [0.0002, 0.0025] | 0.002 | 0.003 | 0.072 | 82.1 | 0.18 | 2,345 | 1.00 |
| 10 | text_hash_tower | 0.0006 [0.0001, 0.0013] | 0.001 | 0.003 | 0.044 | 6.7 | 0.01 | 2,389 | 1.00 |
| 11 | bpr_mf | 0.0004 [0.0001, 0.0008] | 0.001 | 0.003 | 0.006 | 0.3 | 0.01 | 1,420 | 1.00 |
| 12 | hstu | 0.0002 [0.0000, 0.0006] | 0.001 | 0.001 | 0.176 | 15.0 | 0.06 | 2,448 | 1.00 |
| 13 | sasrec | 0.0001 [0.0000, 0.0002] | 0.000 | 0.001 | 0.145 | 11.2 | 0.03 | 2,509 | 1.00 |
| 14 | random | 0.0000 [0.0000, 0.0000] | 0.000 | 0.000 | 0.565 | 0.0 | 0.05 | 1,553 | 0.00 |

### Next-item prediction

Can the method put the user's very next item in its top 10?

| # | Method | Next NDCG@10 [95% CI] | Next HitRate@10 |
|---|---|---|---|
| 1 | ease | 0.0059 [0.0031, 0.0090] | 0.009 |
| 2 | itemknn | 0.0049 [0.0023, 0.0079] | 0.007 |
| 3 | lightgcn | 0.0045 [0.0021, 0.0071] | 0.007 |
| 4 | ials | 0.0038 [0.0015, 0.0065] | 0.005 |
| 5 | xsimgcl | 0.0034 [0.0014, 0.0058] | 0.006 |
| 6 | bert4rec | 0.0015 [0.0000, 0.0030] | 0.002 |
| 7 | dcnv2 | 0.0014 [0.0002, 0.0030] | 0.002 |
| 8 | most_popular | 0.0009 [0.0002, 0.0018] | 0.002 |
| 9 | s3rec | 0.0008 [0.0000, 0.0022] | 0.002 |
| 10 | text_hash_tower | 0.0007 [0.0002, 0.0016] | 0.002 |
| 11 | bpr_mf | 0.0004 [0.0000, 0.0010] | 0.001 |
| 12 | hstu | 0.0001 [0.0000, 0.0006] | 0.001 |
| 13 | random | 0.0000 [0.0000, 0.0000] | 0.000 |
| 14 | sasrec | 0.0000 [0.0000, 0.0000] | 0.000 |

### Beyond accuracy

| Method | Coverage | Gini ↓ | Popularity pct ↓ | Long-tail share | Novelty | ILD | Serendipity | Calibration KL ↓ | Group gap ↓ | Cold-item recall |
|---|---|---|---|---|---|---|---|---|---|---|
| ease | 0.169 | 0.958 | 0.840 | 0.547 | 14.05 | 0.909 | 0.000 | 5.116 | 0.002 | 0.000 |
| itemknn | 0.183 | 0.953 | 0.817 | 0.651 | 14.24 | 0.933 | 0.000 | 5.308 | 0.003 | 0.000 |
| lightgcn | 0.130 | 0.957 | 0.954 | 0.099 | 12.75 | 0.876 | 0.000 | 5.509 | 0.004 | 0.000 |
| ials | 0.067 | 0.983 | 0.986 | 0.006 | 11.89 | 0.903 | 0.000 | 5.510 | 0.002 | 0.000 |
| xsimgcl | 0.090 | 0.979 | 0.965 | 0.088 | 12.00 | 0.916 | 0.000 | 5.778 | 0.003 | 0.000 |
| bert4rec | 0.010 | 0.998 | 0.950 | 0.133 | 11.73 | 0.994 | 0.000 | 6.455 | 0.003 | 0.000 |
| dcnv2 | 0.123 | 0.965 | 0.912 | 0.283 | 12.82 | 0.832 | 0.000 | 6.345 | 0.003 | 0.000 |
| most_popular | 0.001 | 1.000 | 1.000 | 0.000 | 9.84 | 1.000 | 0.000 | 6.457 | 0.001 | 0.000 |
| s3rec | 0.072 | 0.986 | 0.863 | 0.498 | 13.29 | 0.995 | 0.000 | 6.487 | 0.001 | 0.000 |
| text_hash_tower | 0.044 | 0.993 | 0.738 | 0.610 | 14.29 | 0.174 | 0.000 | 6.210 | 0.001 | 0.001 |
| bpr_mf | 0.006 | 1.000 | 0.845 | 0.433 | 14.29 | 0.978 | 0.000 | 6.270 | 0.001 | 0.000 |
| hstu | 0.176 | 0.927 | 0.821 | 0.619 | 14.30 | 0.994 | 0.000 | 6.438 | 0.000 | 0.000 |
| sasrec | 0.145 | 0.944 | 0.834 | 0.573 | 14.11 | 0.994 | 0.000 | 6.451 | 0.000 | 0.000 |
| random | 0.565 | 0.565 | 0.668 | 0.799 | 14.82 | 0.995 | 0.000 | 6.450 | 0.000 | 0.000 |

### Full ranking vs. sampled (1 + 100 negatives)

If the two rank columns disagree, the cheaper sampled protocol would have changed the conclusion.

| Method | Rank (full) | Rank (sampled) | Sampled NDCG@10 |
|---|---|---|---|
| ease | 1 | 9 ↓ | 0.030 |
| itemknn | 2 | 13 ↓ | 0.015 |
| lightgcn | 3 | 5 ↓ | 0.039 |
| ials | 4 | 3 ↑ | 0.047 |
| xsimgcl | 5 | 6 ↓ | 0.038 |
| bert4rec | 6 | 10 ↓ | 0.028 |
| dcnv2 | 7 | 7 | 0.033 |
| most_popular | 8 | 2 ↑ | 0.058 |
| s3rec | 9 | 8 ↑ | 0.033 |
| text_hash_tower | 10 | 1 ↑ | 0.082 |
| bpr_mf | 11 | 11 | 0.021 |
| hstu | 12 | 14 ↓ | 0.014 |
| sasrec | 13 | 12 ↑ | 0.017 |
| random | 14 | 4 ↑ | 0.040 |

### Experimental (not ranked)

| Method | NDCG@10 | Next NDCG@10 |
|---|---|---|
| tiger_lite | 0.0001 [0.0000, 0.0004] | 0.000 |

### Did not run

| Method | Status | Reason |
|---|---|---|
| din | timeout | full-catalog ranking with DIN on a laptop CPU took over 90 minutes (2,000 users x 23,773 items); run DIN on the GPU tier |
| multimodal_tower | unsupported | no item images on disk |

