
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
| 1 | ease ≈ | 0.1061 [0.0947, 0.1171] | 0.102 | 0.444 | 0.009 | 10.2 | 0.45 | 5,579 | 1.00 |
| 2 | puresvd ≈ | 0.0994 [0.0881, 0.1099] | 0.097 | 0.412 | 0.007 | 7.2 | 0.09 | 4,250 | – |
| 3 | slim ≈ | 0.0988 [0.0875, 0.1090] | 0.096 | 0.432 | 0.010 | 224.1 | 0.44 | 4,250 | 1.00 |

### Next-item prediction

Can the method put the user's very next item in its top 10?

| # | Method | Next NDCG@10 [95% CI] | Next HitRate@10 |
|---|---|---|---|
| 1 | puresvd | 0.0140 [0.0088, 0.0203] | 0.029 |
| 2 | ease | 0.0114 [0.0063, 0.0173] | 0.021 |
| 3 | slim | 0.0094 [0.0050, 0.0147] | 0.020 |

### Same users, repeats allowed

Re-listening/re-buying counts as a hit here.

| Method | NDCG@10 (new items only) | NDCG@10 (repeats allowed) |
|---|---|---|
| ease | 0.106 | 0.516 |
| puresvd | 0.099 | 0.456 |
| slim | 0.099 | 0.480 |

### Beyond accuracy

| Method | Coverage | Gini ↓ | Popularity pct ↓ | Long-tail share | Novelty | ILD | Serendipity | Calibration KL ↓ | Group gap ↓ | Cold-item recall |
|---|---|---|---|---|---|---|---|---|---|---|
| ease | 0.009 | 0.997 | 0.995 | 0.003 | 10.76 | – | 0.000 | – | 0.024 | 0.000 |
| puresvd | 0.007 | 0.998 | 0.998 | 0.000 | 10.27 | – | 0.001 | – | 0.039 | 0.000 |
| slim | 0.010 | 0.996 | 0.991 | 0.007 | 11.05 | – | 0.000 | – | 0.024 | 0.000 |

### Full ranking vs. sampled (1 + 100 negatives)

If the two rank columns disagree, the cheaper sampled protocol would have changed the conclusion.

| Method | Rank (full) | Rank (sampled) | Sampled NDCG@10 |
|---|---|---|---|
| ease | 1 | 2 ↓ | 0.473 |
| puresvd | 2 | 1 ↑ | 0.490 |
| slim | 3 | 3 | 0.369 |

