
**Split at a glance**

|  |  |
|---|---|
| Users / items | 1,088 / 22,666 |
| Events before / after the test cutoff | 47,809 / 2,063 |
| Test window starts (UTC) | 2020-09-16 00:00:00 |
| Warm eval users (have history and a new test item) | 577 |
| Cold test users (no history; popularity is their only option) | 3 |
| Share of test interactions that repeat a past item | 4.4% |
| Repeat policies | exclude_seen (primary: exclude_seen) |

### Top-N for warm users (full-catalog ranking)

Sorted by NDCG@10. ≈ marks methods whose 95% confidence interval overlaps the best one's.

| # | Method | NDCG@10 [95% CI] | Recall@10 | HitRate@10 | Coverage@10 | Train s | Score s / 1k users | Peak RSS MB | Personal expl. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | most_popular ≈ | 0.0032 [0.0007, 0.0064] | 0.005 | 0.016 | 0.001 | 0.0 | 0.00 | 659 | 0.00 |
| 2 | ease ≈ | 0.0026 [0.0010, 0.0046] | 0.004 | 0.016 | 0.072 | 50.4 | 0.14 | 7,135 | 1.00 |
| 3 | itemknn ≈ | 0.0026 [0.0005, 0.0056] | 0.002 | 0.009 | 0.161 | 0.3 | 0.04 | 731 | 1.00 |
| 4 | dcnv2 ≈ | 0.0015 [0.0000, 0.0038] | 0.002 | 0.005 | 0.031 | 4.1 | 4.09 | 5,600 | 1.00 |
| 5 | ials ≈ | 0.0015 [0.0004, 0.0028] | 0.002 | 0.010 | 0.047 | 0.3 | 0.01 | 672 | 1.00 |
| 6 | text_hash_tower ≈ | 0.0012 [0.0000, 0.0033] | 0.001 | 0.003 | 0.008 | 22.3 | 0.01 | 3,723 | 1.00 |
| 7 | s3rec ≈ | 0.0012 [0.0001, 0.0027] | 0.001 | 0.007 | 0.062 | 43.1 | 0.10 | 1,416 | 1.00 |
| 8 | bpr_mf ≈ | 0.0010 [0.0001, 0.0021] | 0.001 | 0.007 | 0.043 | 0.4 | 0.01 | 669 | 1.00 |
| 9 | hstu ≈ | 0.0008 [0.0000, 0.0022] | 0.001 | 0.003 | 0.033 | 20.2 | 0.04 | 3,394 | 1.00 |
| 10 | sasrec ≈ | 0.0004 [0.0000, 0.0011] | 0.001 | 0.003 | 0.018 | 19.9 | 0.04 | 3,256 | 1.00 |
| 11 | bert4rec ≈ | 0.0003 [0.0000, 0.0008] | 0.000 | 0.003 | 0.012 | 26.3 | 0.12 | 1,415 | 1.00 |
| 12 | lightgcn ≈ | 0.0003 [0.0000, 0.0009] | 0.000 | 0.002 | 0.132 | 3.8 | 0.01 | 811 | 1.00 |
| 13 | random | 0.0000 [0.0000, 0.0000] | 0.000 | 0.000 | 0.224 | 0.0 | 0.05 | 658 | 0.00 |
| 14 | xsimgcl | 0.0000 [0.0000, 0.0000] | 0.000 | 0.000 | 0.120 | 6.0 | 0.01 | 1,070 | 1.00 |

### Next-item prediction

Can the method put the user's very next item in its top 10?

| # | Method | Next NDCG@10 [95% CI] | Next HitRate@10 |
|---|---|---|---|
| 1 | ials | 0.0012 [0.0000, 0.0031] | 0.003 |
| 2 | most_popular | 0.0011 [0.0000, 0.0033] | 0.002 |
| 3 | s3rec | 0.0009 [0.0000, 0.0026] | 0.002 |
| 4 | dcnv2 | 0.0009 [0.0000, 0.0026] | 0.002 |
| 5 | text_hash_tower | 0.0007 [0.0000, 0.0022] | 0.002 |
| 6 | bert4rec | 0.0006 [0.0000, 0.0019] | 0.002 |
| 7 | sasrec | 0.0005 [0.0000, 0.0016] | 0.002 |
| 8 | ease | 0.0005 [0.0000, 0.0020] | 0.002 |
| 9 | random | 0.0000 [0.0000, 0.0000] | 0.000 |
| 10 | itemknn | 0.0000 [0.0000, 0.0000] | 0.000 |
| 11 | bpr_mf | 0.0000 [0.0000, 0.0000] | 0.000 |
| 12 | lightgcn | 0.0000 [0.0000, 0.0000] | 0.000 |
| 13 | xsimgcl | 0.0000 [0.0000, 0.0000] | 0.000 |
| 14 | hstu | 0.0000 [0.0000, 0.0000] | 0.000 |

### Beyond accuracy

| Method | Coverage | Gini ↓ | Popularity pct ↓ | Long-tail share | Novelty | ILD | Serendipity | Calibration KL ↓ | Group gap ↓ | Cold-item recall |
|---|---|---|---|---|---|---|---|---|---|---|
| most_popular | 0.001 | 1.000 | 0.982 | 0.000 | 12.41 | 0.668 | 0.000 | 4.743 | 0.005 | 0.000 |
| ease | 0.072 | 0.974 | 0.965 | 0.073 | 12.29 | 0.829 | 0.000 | 3.783 | 0.003 | 0.000 |
| itemknn | 0.161 | 0.889 | 0.737 | 0.614 | 14.09 | 0.890 | 0.000 | 3.751 | 0.004 | 0.000 |
| dcnv2 | 0.031 | 0.992 | 0.919 | 0.202 | 12.74 | 0.764 | 0.000 | 4.873 | 0.002 | 0.000 |
| ials | 0.047 | 0.983 | 0.989 | 0.002 | 12.15 | 0.856 | 0.000 | 3.886 | 0.001 | 0.000 |
| text_hash_tower | 0.008 | 0.997 | 0.903 | 0.213 | 13.00 | 0.558 | 0.000 | 5.100 | 0.002 | 0.000 |
| s3rec | 0.062 | 0.979 | 0.806 | 0.462 | 13.10 | 0.920 | 0.000 | 4.349 | 0.002 | 0.000 |
| bpr_mf | 0.043 | 0.987 | 0.941 | 0.139 | 13.01 | 0.904 | 0.000 | 3.905 | 0.002 | 0.000 |
| hstu | 0.033 | 0.992 | 0.765 | 0.559 | 13.97 | 0.947 | 0.000 | 4.168 | 0.004 | 0.000 |
| sasrec | 0.018 | 0.996 | 0.785 | 0.559 | 13.95 | 0.956 | 0.000 | 4.098 | 0.001 | 0.000 |
| bert4rec | 0.012 | 0.997 | 0.908 | 0.164 | 13.00 | 0.895 | 0.000 | 4.003 | 0.001 | 0.000 |
| lightgcn | 0.132 | 0.920 | 0.839 | 0.420 | 13.70 | 0.898 | 0.000 | 4.060 | 0.001 | 0.000 |
| random | 0.224 | 0.800 | 0.661 | 0.803 | 14.67 | 0.942 | 0.000 | 4.232 | 0.000 | 0.000 |
| xsimgcl | 0.120 | 0.929 | 0.849 | 0.421 | 13.90 | 0.906 | 0.000 | 4.109 | 0.000 | 0.000 |

### Full ranking vs. sampled (1 + 100 negatives)

If the two rank columns disagree, the cheaper sampled protocol would have changed the conclusion.

| Method | Rank (full) | Rank (sampled) | Sampled NDCG@10 |
|---|---|---|---|
| most_popular | 1 | 1 | 0.181 |
| ease | 2 | 2 | 0.059 |
| itemknn | 3 | 14 ↓ | 0.024 |
| dcnv2 | 4 | 8 ↓ | 0.043 |
| ials | 5 | 3 ↑ | 0.054 |
| text_hash_tower | 6 | 7 ↓ | 0.044 |
| s3rec | 7 | 5 ↑ | 0.045 |
| bpr_mf | 8 | 9 ↓ | 0.042 |
| hstu | 9 | 12 ↓ | 0.024 |
| sasrec | 10 | 10 | 0.030 |
| bert4rec | 11 | 4 ↑ | 0.048 |
| lightgcn | 12 | 11 ↑ | 0.027 |
| random | 13 | 6 ↑ | 0.044 |
| xsimgcl | 14 | 13 ↑ | 0.024 |

### Experimental (not ranked)

| Method | NDCG@10 | Next NDCG@10 |
|---|---|---|
| tiger_lite | 0.0000 [0.0000, 0.0000] | 0.000 |

### Did not run

| Method | Status | Reason |
|---|---|---|
| multimodal_tower | unsupported | no item images on disk |

