
**Split at a glance**

|  |  |
|---|---|
| Users / items | 175 / 7,578 |
| Events before / after the test cutoff | 44,229 / 5,418 |
| Test window starts (UTC) | 2009-02-02 16:02:09 |
| Warm eval users (have history and a new test item) | 117 |
| Cold test users (no history; popularity is their only option) | 21 |
| Share of test interactions that repeat a past item | 36.8% |
| Repeat policies | exclude_seen, allow_repeats (primary: exclude_seen) |

### Top-N for warm users (full-catalog ranking)

Sorted by NDCG@10. ≈ marks methods whose 95% confidence interval overlaps the best one's.

| # | Method | NDCG@10 [95% CI] | Recall@10 | HitRate@10 | Coverage@10 | Train s | Score s / 1k users | Peak RSS MB | Personal expl. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ease ≈ | 0.0191 [0.0093, 0.0308] | 0.017 | 0.137 | 0.026 | 2.2 | 0.09 | 2,072 | 1.00 |
| 2 | ials ≈ | 0.0189 [0.0090, 0.0315] | 0.025 | 0.120 | 0.066 | 0.1 | 0.00 | 387 | 1.00 |
| 3 | itemknn ≈ | 0.0147 [0.0065, 0.0247] | 0.013 | 0.094 | 0.030 | 0.1 | 0.05 | 414 | 1.00 |
| 4 | lightgcn ≈ | 0.0117 [0.0041, 0.0205] | 0.012 | 0.077 | 0.062 | 2.2 | 0.00 | 510 | 1.00 |
| 5 | bpr_mf ≈ | 0.0117 [0.0043, 0.0220] | 0.012 | 0.077 | 0.058 | 0.2 | 0.00 | 385 | 1.00 |
| 6 | hstu ≈ | 0.0082 [0.0006, 0.0189] | 0.019 | 0.034 | 0.074 | 17.8 | 0.04 | 3,676 | 1.00 |
| 7 | dcnv2 ≈ | 0.0080 [0.0026, 0.0151] | 0.006 | 0.060 | 0.024 | 3.7 | 1.68 | 4,059 | 1.00 |
| 8 | most_popular ≈ | 0.0075 [0.0012, 0.0157] | 0.011 | 0.034 | 0.002 | 0.0 | 0.00 | 380 | 0.00 |
| 9 | bert4rec ≈ | 0.0073 [0.0026, 0.0133] | 0.008 | 0.068 | 0.014 | 19.9 | 0.05 | 859 | 1.00 |
| 10 | text_hash_tower ≈ | 0.0068 [0.0023, 0.0127] | 0.008 | 0.060 | 0.008 | 10.7 | 0.01 | 3,612 | 1.00 |
| 11 | sasrec | 0.0030 [0.0006, 0.0063] | 0.004 | 0.034 | 0.065 | 16.8 | 0.04 | 3,687 | 1.00 |
| 12 | random | 0.0026 [0.0000, 0.0075] | 0.002 | 0.017 | 0.141 | 0.0 | 0.02 | 380 | 0.00 |
| 13 | xsimgcl | 0.0000 [0.0000, 0.0000] | 0.000 | 0.000 | 0.065 | 3.5 | 0.00 | 565 | 1.00 |

### Next-item prediction

Can the method put the user's very next item in its top 10?

| # | Method | Next NDCG@10 [95% CI] | Next HitRate@10 |
|---|---|---|---|
| 1 | ials | 0.0116 [0.0000, 0.0264] | 0.026 |
| 2 | bpr_mf | 0.0085 [0.0000, 0.0256] | 0.009 |
| 3 | hstu | 0.0070 [0.0000, 0.0177] | 0.017 |
| 4 | itemknn | 0.0054 [0.0000, 0.0162] | 0.009 |
| 5 | ease | 0.0054 [0.0000, 0.0162] | 0.009 |
| 6 | text_hash_tower | 0.0030 [0.0000, 0.0091] | 0.009 |
| 7 | most_popular | 0.0026 [0.0000, 0.0077] | 0.009 |
| 8 | dcnv2 | 0.0025 [0.0000, 0.0074] | 0.009 |
| 9 | random | 0.0000 [0.0000, 0.0000] | 0.000 |
| 10 | lightgcn | 0.0000 [0.0000, 0.0000] | 0.000 |
| 11 | xsimgcl | 0.0000 [0.0000, 0.0000] | 0.000 |
| 12 | sasrec | 0.0000 [0.0000, 0.0000] | 0.000 |
| 13 | bert4rec | 0.0000 [0.0000, 0.0000] | 0.000 |

### Same users, repeats allowed

Re-listening/re-buying counts as a hit here.

| Method | NDCG@10 (new items only) | NDCG@10 (repeats allowed) |
|---|---|---|
| ease | 0.019 | 0.131 |
| ials | 0.019 | 0.185 |
| itemknn | 0.015 | 0.113 |
| lightgcn | 0.012 | 0.152 |
| bpr_mf | 0.012 | 0.094 |
| hstu | 0.008 | 0.080 |
| dcnv2 | 0.008 | 0.048 |
| most_popular | 0.008 | 0.025 |
| bert4rec | 0.007 | 0.037 |
| text_hash_tower | 0.007 | 0.030 |
| sasrec | 0.003 | 0.076 |
| random | 0.003 | 0.004 |
| xsimgcl | 0.000 | 0.072 |

### Beyond accuracy

| Method | Coverage | Gini ↓ | Popularity pct ↓ | Long-tail share | Novelty | ILD | Serendipity | Calibration KL ↓ | Group gap ↓ | Cold-item recall |
|---|---|---|---|---|---|---|---|---|---|---|
| ease | 0.026 | 0.991 | 0.981 | 0.015 | 9.17 | – | 0.003 | – | – | 0.000 |
| ials | 0.066 | 0.962 | 0.954 | 0.029 | 10.35 | – | 0.003 | – | – | 0.000 |
| itemknn | 0.030 | 0.990 | 0.969 | 0.041 | 9.36 | – | 0.001 | – | – | 0.000 |
| lightgcn | 0.062 | 0.971 | 0.914 | 0.167 | 10.41 | – | 0.003 | – | – | 0.000 |
| bpr_mf | 0.058 | 0.967 | 0.937 | 0.085 | 10.68 | – | 0.003 | – | – | 0.000 |
| hstu | 0.074 | 0.958 | 0.872 | 0.264 | 11.10 | – | 0.000 | – | – | 0.000 |
| dcnv2 | 0.024 | 0.992 | 0.977 | 0.025 | 9.07 | – | 0.000 | – | – | 0.000 |
| most_popular | 0.002 | 0.999 | 0.998 | 0.000 | 8.13 | – | 0.000 | – | – | 0.000 |
| bert4rec | 0.014 | 0.993 | 0.986 | 0.012 | 8.96 | – | 0.000 | – | – | 0.000 |
| text_hash_tower | 0.008 | 0.997 | 0.969 | 0.032 | 8.86 | – | 0.000 | – | – | 0.000 |
| sasrec | 0.065 | 0.965 | 0.870 | 0.263 | 11.16 | – | 0.001 | – | – | 0.000 |
| random | 0.141 | 0.869 | 0.615 | 0.797 | 13.85 | – | 0.002 | – | – | 0.003 |
| xsimgcl | 0.065 | 0.967 | 0.752 | 0.568 | 12.82 | – | 0.000 | – | – | 0.000 |

### Full ranking vs. sampled (1 + 100 negatives)

If the two rank columns disagree, the cheaper sampled protocol would have changed the conclusion.

| Method | Rank (full) | Rank (sampled) | Sampled NDCG@10 |
|---|---|---|---|
| ease | 1 | 1 | 0.232 |
| ials | 2 | 5 ↓ | 0.140 |
| itemknn | 3 | 3 | 0.191 |
| lightgcn | 4 | 4 | 0.143 |
| bpr_mf | 5 | 2 ↑ | 0.195 |
| hstu | 6 | 10 ↓ | 0.067 |
| dcnv2 | 7 | 7 | 0.127 |
| most_popular | 8 | 8 | 0.119 |
| bert4rec | 9 | 6 ↑ | 0.136 |
| text_hash_tower | 10 | 9 ↑ | 0.078 |
| sasrec | 11 | 11 | 0.058 |
| random | 12 | 12 | 0.052 |
| xsimgcl | 13 | 13 | 0.025 |

### Experimental (not ranked)

| Method | NDCG@10 | Next NDCG@10 |
|---|---|---|
| tiger_lite | 0.0010 [0.0000, 0.0030] | 0.000 |

### Did not run

| Method | Status | Reason |
|---|---|---|
| s3rec | unsupported | S3-Rec needs item attributes; this dataset has fewer than 2 category values |
| multimodal_tower | unsupported | no item images on disk |

