
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
| 1 | most_popular ≈ | 0.0531 [0.0502, 0.0562] | 0.106 | 0.147 | 0.002 | 0.1 | 0.01 | 2,251 | 0.00 |
| 2 | sasrec ≈ | 0.0519 [0.0488, 0.0552] | 0.091 | 0.130 | 0.073 | 1,748 | 0.08 | 3,072 | 1.00 |
| 3 | text_hash_tower | 0.0451 [0.0424, 0.0478] | 0.078 | 0.111 | 0.103 | 1,426 | 0.04 | 2,954 | 1.00 |
| 4 | ease | 0.0417 [0.0389, 0.0443] | 0.074 | 0.106 | 0.170 | 28.9 | 0.11 | 8,278 | 1.00 |
| 5 | itemknn | 0.0322 [0.0297, 0.0347] | 0.056 | 0.082 | 0.387 | 5.2 | 0.05 | 2,221 | 1.00 |
| 6 | dcnv2 | 0.0259 [0.0238, 0.0282] | 0.048 | 0.069 | 0.218 | 1,174 | 1.50 | 2,946 | 1.00 |
| 7 | ials | 0.0229 [0.0208, 0.0250] | 0.039 | 0.060 | 0.069 | 336.9 | 0.01 | 3,893 | 1.00 |
| 8 | bpr_mf | 0.0195 [0.0176, 0.0215] | 0.035 | 0.051 | 0.332 | 214.1 | 0.01 | 3,916 | 1.00 |
| 9 | random | 0.0003 [0.0001, 0.0006] | 0.000 | 0.001 | 0.999 | 0.0 | 0.07 | 2,260 | 0.00 |

### Next-item prediction

Can the method put the user's very next item in its top 10?

| # | Method | Next NDCG@10 [95% CI] | Next HitRate@10 |
|---|---|---|---|
| 1 | most_popular | 0.0516 [0.0486, 0.0550] | 0.111 |
| 2 | sasrec | 0.0482 [0.0451, 0.0517] | 0.092 |
| 3 | text_hash_tower | 0.0423 [0.0394, 0.0452] | 0.080 |
| 4 | ease | 0.0387 [0.0358, 0.0415] | 0.075 |
| 5 | itemknn | 0.0299 [0.0274, 0.0326] | 0.057 |
| 6 | dcnv2 | 0.0244 [0.0222, 0.0268] | 0.049 |
| 7 | ials | 0.0205 [0.0183, 0.0226] | 0.038 |
| 8 | bpr_mf | 0.0187 [0.0167, 0.0208] | 0.037 |
| 9 | random | 0.0003 [0.0001, 0.0007] | 0.001 |

### Beyond accuracy

| Method | Coverage | Gini ↓ | Popularity pct ↓ | Long-tail share | Novelty | ILD | Serendipity | Calibration KL ↓ | Group gap ↓ | Cold-item recall |
|---|---|---|---|---|---|---|---|---|---|---|
| most_popular | 0.002 | 0.999 | 0.989 | 0.000 | 8.56 | 0.795 | 0.000 | 1.395 | 0.004 | 0.000 |
| sasrec | 0.073 | 0.991 | 0.990 | 0.004 | 8.57 | 0.675 | 0.000 | 1.033 | 0.007 | 0.000 |
| text_hash_tower | 0.103 | 0.990 | 0.990 | 0.006 | 8.29 | 0.674 | 0.000 | 0.840 | 0.012 | 0.000 |
| ease | 0.170 | 0.978 | 0.985 | 0.010 | 8.83 | 0.639 | 0.000 | 0.844 | 0.011 | 0.000 |
| itemknn | 0.387 | 0.948 | 0.920 | 0.121 | 9.95 | 0.594 | 0.000 | 1.039 | 0.011 | 0.000 |
| dcnv2 | 0.218 | 0.982 | 0.985 | 0.016 | 8.34 | 0.717 | 0.000 | 1.488 | 0.005 | 0.000 |
| ials | 0.069 | 0.973 | 0.985 | 0.000 | 9.75 | 0.668 | 0.000 | 0.957 | 0.004 | 0.000 |
| bpr_mf | 0.332 | 0.881 | 0.866 | 0.290 | 12.99 | 0.590 | 0.000 | 1.090 | 0.006 | 0.000 |
| random | 0.999 | 0.220 | 0.510 | 0.798 | 17.69 | 0.750 | 0.000 | 1.849 | 0.000 | 0.000 |

### Full ranking vs. sampled (1 + 100 negatives)

If the two rank columns disagree, the cheaper sampled protocol would have changed the conclusion.

| Method | Rank (full) | Rank (sampled) | Sampled NDCG@10 |
|---|---|---|---|
| most_popular | 1 | 1 | 0.521 |
| sasrec | 2 | 2 | 0.509 |
| text_hash_tower | 3 | 3 | 0.486 |
| ease | 4 | 4 | 0.424 |
| itemknn | 5 | 7 ↓ | 0.267 |
| dcnv2 | 6 | 5 ↑ | 0.404 |
| ials | 7 | 6 ↑ | 0.323 |
| bpr_mf | 8 | 8 | 0.231 |
| random | 9 | 9 | 0.044 |

### Experimental (not ranked)

| Method | NDCG@10 | Next NDCG@10 |
|---|---|---|
| tiger_lite | 0.0487 [0.0457, 0.0519] | 0.045 |

### Did not run

| Method | Status | Reason |
|---|---|---|
| multimodal_tower | unsupported | no item images on disk |

