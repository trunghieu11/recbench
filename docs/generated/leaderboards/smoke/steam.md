
**Split at a glance**

|  |  |
|---|---|
| Users / items | 8,454 / 5,543 |
| Events before / after the test cutoff | 41,978 / 7,693 |
| Test window starts (UTC) | 2017-10-17 00:00:00 |
| Warm eval users (have history and a new test item) | 3,896 |
| Cold test users (no history; popularity is their only option) | 450 |
| Share of test interactions that repeat a past item | 2.1% |
| Repeat policies | exclude_seen (primary: exclude_seen) |

### Top-N for warm users (full-catalog ranking)

Sorted by NDCG@10. ≈ marks methods whose 95% confidence interval overlaps the best one's.

| # | Method | NDCG@10 [95% CI] | Recall@10 | HitRate@10 | Coverage@10 | Train s | Score s / 1k users | Peak RSS MB | Personal expl. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | most_popular ≈ | 0.0455 [0.0394, 0.0521] | 0.090 | 0.118 | 0.005 | 0.0 | 0.00 | 678 | 0.00 |
| 2 | s3rec ≈ | 0.0410 [0.0352, 0.0474] | 0.076 | 0.104 | 0.016 | 38.4 | 0.07 | 1,542 | 1.00 |
| 3 | lightgcn ≈ | 0.0388 [0.0328, 0.0447] | 0.074 | 0.101 | 0.025 | 3.2 | 0.00 | 806 | 1.00 |
| 4 | bert4rec ≈ | 0.0357 [0.0303, 0.0413] | 0.067 | 0.093 | 0.007 | 18.6 | 0.05 | 1,318 | 1.00 |
| 5 | text_hash_tower ≈ | 0.0355 [0.0299, 0.0414] | 0.064 | 0.089 | 0.009 | 5.5 | 0.00 | 1,119 | 1.00 |
| 6 | dcnv2 ≈ | 0.0341 [0.0284, 0.0398] | 0.064 | 0.091 | 0.014 | 3.8 | 1.02 | 5,401 | 1.00 |
| 7 | ease | 0.0307 [0.0256, 0.0363] | 0.053 | 0.077 | 0.261 | 0.9 | 0.03 | 1,505 | 1.00 |
| 8 | xsimgcl | 0.0257 [0.0209, 0.0306] | 0.044 | 0.065 | 0.099 | 5.2 | 0.00 | 961 | 1.00 |
| 9 | ials | 0.0238 [0.0190, 0.0287] | 0.041 | 0.064 | 0.078 | 0.2 | 0.00 | 671 | 1.00 |
| 10 | itemknn | 0.0237 [0.0190, 0.0289] | 0.041 | 0.064 | 0.462 | 0.1 | 0.01 | 676 | 1.00 |
| 11 | sasrec | 0.0237 [0.0190, 0.0287] | 0.044 | 0.059 | 0.045 | 8.4 | 0.02 | 1,388 | 1.00 |
| 12 | hstu | 0.0173 [0.0132, 0.0217] | 0.028 | 0.044 | 0.078 | 10.4 | 0.03 | 1,318 | 1.00 |
| 13 | bpr_mf | 0.0125 [0.0094, 0.0164] | 0.021 | 0.035 | 0.083 | 0.4 | 0.00 | 674 | 1.00 |
| 14 | random | 0.0012 [0.0004, 0.0022] | 0.002 | 0.004 | 0.973 | 0.0 | 0.02 | 684 | 0.00 |

### Next-item prediction

Can the method put the user's very next item in its top 10?

| # | Method | Next NDCG@10 [95% CI] | Next HitRate@10 |
|---|---|---|---|
| 1 | most_popular | 0.0448 [0.0383, 0.0517] | 0.092 |
| 2 | s3rec | 0.0388 [0.0330, 0.0452] | 0.079 |
| 3 | lightgcn | 0.0366 [0.0305, 0.0431] | 0.075 |
| 4 | text_hash_tower | 0.0336 [0.0278, 0.0396] | 0.065 |
| 5 | bert4rec | 0.0334 [0.0274, 0.0393] | 0.068 |
| 6 | dcnv2 | 0.0321 [0.0261, 0.0381] | 0.064 |
| 7 | ease | 0.0291 [0.0237, 0.0351] | 0.054 |
| 8 | xsimgcl | 0.0245 [0.0192, 0.0297] | 0.046 |
| 9 | itemknn | 0.0226 [0.0178, 0.0281] | 0.043 |
| 10 | sasrec | 0.0226 [0.0175, 0.0281] | 0.044 |
| 11 | ials | 0.0220 [0.0171, 0.0273] | 0.043 |
| 12 | hstu | 0.0172 [0.0126, 0.0219] | 0.029 |
| 13 | bpr_mf | 0.0111 [0.0078, 0.0151] | 0.022 |
| 14 | random | 0.0012 [0.0003, 0.0022] | 0.003 |

### Beyond accuracy

| Method | Coverage | Gini ↓ | Popularity pct ↓ | Long-tail share | Novelty | ILD | Serendipity | Calibration KL ↓ | Group gap ↓ | Cold-item recall |
|---|---|---|---|---|---|---|---|---|---|---|
| most_popular | 0.005 | 0.998 | 0.970 | 0.000 | 9.05 | 0.702 | 0.000 | 1.512 | 0.012 | 0.000 |
| s3rec | 0.016 | 0.997 | 0.998 | 0.001 | 7.28 | 0.688 | 0.000 | 1.588 | 0.016 | 0.000 |
| lightgcn | 0.025 | 0.998 | 0.975 | 0.054 | 7.46 | 0.721 | 0.000 | 1.744 | 0.019 | 0.000 |
| bert4rec | 0.007 | 0.998 | 0.999 | 0.000 | 7.17 | 0.714 | 0.000 | 1.733 | 0.019 | 0.000 |
| text_hash_tower | 0.009 | 0.998 | 0.997 | 0.001 | 7.41 | 0.687 | 0.000 | 1.583 | 0.018 | 0.000 |
| dcnv2 | 0.014 | 0.997 | 0.998 | 0.000 | 7.33 | 0.745 | 0.000 | 1.693 | 0.016 | 0.000 |
| ease | 0.261 | 0.965 | 0.971 | 0.046 | 8.41 | 0.708 | 0.000 | 1.246 | 0.007 | 0.000 |
| xsimgcl | 0.099 | 0.981 | 0.985 | 0.008 | 8.49 | 0.741 | 0.000 | 1.365 | 0.004 | 0.000 |
| ials | 0.078 | 0.973 | 0.986 | 0.000 | 8.79 | 0.714 | 0.000 | 1.218 | 0.011 | 0.000 |
| itemknn | 0.462 | 0.887 | 0.897 | 0.194 | 9.89 | 0.715 | 0.000 | 1.225 | 0.012 | 0.000 |
| sasrec | 0.045 | 0.994 | 0.963 | 0.065 | 8.42 | 0.762 | 0.000 | 1.447 | 0.015 | 0.000 |
| hstu | 0.078 | 0.988 | 0.919 | 0.181 | 9.44 | 0.728 | 0.000 | 1.659 | 0.002 | 0.000 |
| bpr_mf | 0.083 | 0.978 | 0.917 | 0.101 | 10.58 | 0.712 | 0.000 | 1.363 | 0.007 | 0.000 |
| random | 0.973 | 0.296 | 0.587 | 0.802 | 13.60 | 0.770 | 0.000 | 1.681 | 0.001 | 0.004 |

### Full ranking vs. sampled (1 + 100 negatives)

If the two rank columns disagree, the cheaper sampled protocol would have changed the conclusion.

| Method | Rank (full) | Rank (sampled) | Sampled NDCG@10 |
|---|---|---|---|
| most_popular | 1 | 1 | 0.333 |
| s3rec | 2 | 2 | 0.313 |
| lightgcn | 3 | 3 | 0.312 |
| bert4rec | 4 | 6 ↓ | 0.292 |
| text_hash_tower | 5 | 4 ↑ | 0.308 |
| dcnv2 | 6 | 5 ↑ | 0.303 |
| ease | 7 | 7 | 0.233 |
| xsimgcl | 8 | 9 ↓ | 0.162 |
| ials | 9 | 8 ↑ | 0.213 |
| itemknn | 10 | 11 ↓ | 0.156 |
| sasrec | 11 | 10 ↑ | 0.161 |
| hstu | 12 | 12 | 0.134 |
| bpr_mf | 13 | 13 | 0.115 |
| random | 14 | 14 | 0.043 |

### Experimental (not ranked)

| Method | NDCG@10 | Next NDCG@10 |
|---|---|---|
| tiger_lite | 0.0202 [0.0158, 0.0250] | 0.019 |

### Did not run

| Method | Status | Reason |
|---|---|---|
| multimodal_tower | unsupported | no item images on disk |

