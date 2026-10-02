# Data pipeline and file formats

```mermaid
flowchart LR
    D[download] --> C[to_clean]
    C --> M[materialize]
    M --> F[split files]
```

Command: `python -m recbench.pipeline.prepare --config <benchmark.yaml> [--datasets a,b] [--tier smoke]`
(`src/recbench/pipeline/prepare.py::prepare_one`).

## 1. Download and clean (adapters)

Each adapter in `src/recbench/datasets/` has:

- `spec`: a `DatasetSpec` (name, domain, feedback, `split_rule`, `test_days`, `repeat_policies`);
- `download(raw_dir)`: fetches files into `data/raw/<name>/` (skipped if already there);
- `to_clean(raw_dir, clean_dir)`: writes `interactions.parquet`, `items.parquet`, `users.parquet` to
  `data/clean/<name>/`, following the contract in `src/recbench/schema.py`;
- `CLEAN_VERSION`: stored in `data/clean/<name>/_clean_version.txt`. Cleaning is re-run only when it changes.

If a dataset fails, the traceback goes to `data/unavailable/<name>-<tier>.txt` and other datasets continue.

## 2. Materialize (the split)

`src/recbench/pipeline/materialize.py::materialize`, step by step:

1. **Read events** with DuckDB (time zone forced to UTC). Each event gets `ts_us` (integer microseconds since
   1970, UTC) and `event_seq` (its row number in the clean file, used to break ties).
2. **Cutoffs on the full data** (`src/recbench/pipeline/materialize.py::cutoffs`):
    - `quantile`: `test_start` = 90th percentile of event times; `valid_start` = 80th percentile.
    - `last_days` (H&M): test = the last 7 calendar days; validation = the 7 days before.
3. **Tier sampling** (`TIERS` in the same file):

    | Tier | Events | Per-user caps | Eval users (min / max) |
    |---|---|---|---|
    | smoke | ~50,000 | 300 most recent pre-test, first 50 test | 30 / 10,000 |
    | standard | ~1,000,000 | 1,000 / 200 | 200 / 10,000 |
    | slice | ~40,000, ≤ 10,000 items | 200 / 50 | 100 / 1,000 |
    | full | all | none | 500 / 10,000 |

    Users are sampled with a fixed seed: 80% of the event budget goes to users who have both pre-test and test
    events. Because the cutoffs come from the full data, a smoke split is a true subset of the full one.
4. **Sessions:** a gap of more than 30 minutes starts a new session (unless the source has session ids).
5. **Ids:** `user_idx` and `item_idx` are assigned in sorted id order, starting at **1**. Index 0 is padding.
6. **Parts:** `train` (before `valid_start`), `valid` (until `test_start`), `test` (after). "Pre-test" = train + valid.
7. **Outputs** (below), then a guard: fewer warm evaluation users than the tier's minimum raises `SplitError`.

## 3. The split files (`data/splits/<dataset>/<tier>/`)

### `train.parquet`, `valid.parquet`, `test.parquet` (one row per event)

| Column | Type | Meaning |
|---|---|---|
| user_id, item_id | string | original ids |
| user_idx, item_idx | int32 | integer ids (from 1) |
| timestamp | timestamp | UTC time (for people) |
| ts_us | int64 | UTC microseconds (for code) |
| event_seq | int64 | tie-break order from the clean file |
| session_id | string | explicit or derived session |
| feedback_type, value | string, double | as in the clean table |
| is_cold_user, is_cold_item | bool | no pre-test events for this user / item |

### Pre-test arrays (what `TrainView` reads)

| File | Shape | Meaning |
|---|---|---|
| `pretest_offsets.npy` | int64 [n_users + 2] | user u's events are positions `offsets[u]` to `offsets[u+1]` |
| `pretest_items.npy` | int32 [pre-test events] | item index of each event, ordered by (user, time, event_seq) |
| `pretest_ts.npy` | int64 [pre-test events] | event time in microseconds |

This is the CSR layout explained in [the interaction matrix](../dictionary/concepts/interaction-matrix.md).

### `items.parquet` and `users.parquet`

| File | Columns |
|---|---|
| items | item_idx, item_id, text, category, image_path, pretest_count, recent_count (last 28 days), first_ts_us, is_cold |
| users | user_idx, user_id, attributes, n_pretest, group_label (cold / light / medium / heavy) |

### Evaluation side (read by `EvalSplit`, never by models)

| File | Columns | Meaning |
|---|---|---|
| `test_items.parquet` | user_idx, item_idx, first_ts_us, test_rank, is_repeat, is_cold_item | the first occurrence of each (user, item) in the test window |
| `eval_users.parquet` | user_idx, is_warm, next_item (+ `next_item__allow_repeats` for Last.fm) | sampled evaluation users and their next relevant item |
| `candidates.parquet` | user_idx, item_idx, label | secondary protocol: the next item (label 1) + 100 unseen warm items |

### `meta.json`

Identity (`dataset`, `tier`, `contract_version` = 2, `protocol_version`, `split_hash`), cutoffs (`valid_start`,
`test_start` as text and as `*_us` integers), counts (`n_users`, `n_items`, `n_pretest`, `n_train`, `n_valid`,
`n_test`, `n_test_users`, `n_warm_test_users`, `n_cold_test_users`, `n_eval_warm`, `n_eval_cold`, `n_cold_items`),
`repeat_share`, `repeat_policies`, `primary_policy`, `n_negatives`, `seed`, and the tier parameters. The
`split_hash` changes whenever any of these change; it is part of every run's `config_hash`, and per-split caches
live under `cache/<split_hash>/`.

## Guarantees and their tests

| Guarantee | Test |
|---|---|
| identical cutoffs in any time zone | `tests/test_split.py::test_cutoffs_do_not_depend_on_the_time_zone` |
| pre-test events end before the cutoff | `tests/test_split.py::test_events_respect_the_cutoff` |
| rebuilding gives identical files | `tests/test_split.py::test_rebuilding_gives_identical_files` |
| ties ordered by file position | `tests/test_split.py::test_ties_are_broken_by_file_order` |
| too few eval users fails | `tests/test_split.py::test_too_few_eval_users_fails_loudly` |
