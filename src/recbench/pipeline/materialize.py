"""Turn clean interaction tables into an evaluation split (contract v2).

Steps, in order:
1. Read the clean events and give each one a deterministic tiebreak (its row
   number in the clean file), so equal timestamps always sort the same way.
2. Compute the cutoffs on the FULL dataset, in UTC microseconds:
   valid_start <= test_start. Events before test_start are "pre-test".
3. For smoke/standard/slice tiers, sample users (not rows) so that each kept
   user keeps a complete recent history; the cutoffs stay the same, so a smoke
   split is a subset of the full split.
4. Write train/valid/test parquet files, pre-test history arrays for models,
   item/user tables, test relevance sets, eval users, and the sampled
   1-positive + 100-negative candidate file (secondary protocol).

Nothing here depends on the machine's time zone: DuckDB runs in UTC and all
cutoff arithmetic uses integer microseconds.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from recbench.data import SplitError
from recbench.protocol import N_NEGATIVES, PROTOCOL_VERSION
from recbench.schema import INTERACTION_COLUMNS, ITEM_COLUMNS, USER_COLUMNS, assert_columns

CONTRACT_VERSION = "2"
DAY_US = 86_400_000_000
RECENT_DAYS = 28
SESSION_GAP_US = 30 * 60 * 1_000_000

# target_events: approximate size of a sampled tier; max_user_*: per-user caps
# (the most recent pre-test events and the first test events are kept).
TIERS: dict[str, dict[str, Any]] = {
    "smoke": {"target_events": 50_000, "max_user_pretest": 300, "max_user_test": 50, "min_eval_users": 30, "max_eval_users": 10_000},
    "standard": {"target_events": 1_000_000, "max_user_pretest": 1_000, "max_user_test": 200, "min_eval_users": 200, "max_eval_users": 10_000},
    # Sized for a managed service's free plan: <= 10K items, ~40K events, 1K eval users.
    "slice": {"target_events": 40_000, "max_user_pretest": 200, "max_user_test": 50, "max_items": 10_000, "min_eval_users": 100, "max_eval_users": 1_000},
    "full": {"min_eval_users": 500, "max_eval_users": 10_000},
}

SPLIT_FILES = (
    "meta.json",
    "train.parquet",
    "valid.parquet",
    "test.parquet",
    "items.parquet",
    "users.parquet",
    "test_items.parquet",
    "eval_users.parquet",
    "candidates.parquet",
    "pretest_offsets.npy",
    "pretest_items.npy",
    "pretest_ts.npy",
)


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.execute("PRAGMA threads=4")
    return con


def _iso(us: int) -> str:
    return datetime.fromtimestamp(us / 1_000_000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _clear(out_dir: Path) -> None:
    for name in SPLIT_FILES:
        (out_dir / name).unlink(missing_ok=True)
    for stale in ("sequences.parquet", "targets.parquet", "_train_hist.parquet"):
        (out_dir / stale).unlink(missing_ok=True)
    for path in out_dir.glob("history_*.npy"):
        path.unlink()
    for folder in ("cache", "recbole"):
        shutil.rmtree(out_dir / folder, ignore_errors=True)


def cutoffs(con: duckdb.DuckDBPyConnection, split_rule: str, test_days: int) -> tuple[int, int]:
    """Return (valid_start_us, test_start_us) computed on the whole event table `ev`."""
    if split_rule == "last_days":
        max_us = int(con.execute("SELECT max(ts_us) FROM ev").fetchone()[0])
        last_day_start = max_us - (max_us % DAY_US)
        test_start = last_day_start - (test_days - 1) * DAY_US
        return test_start - test_days * DAY_US, test_start
    test_start = int(con.execute("SELECT quantile_disc(ts_us, 0.9) FROM ev").fetchone()[0])
    valid_start = int(con.execute("SELECT quantile_disc(ts_us, 0.8) FROM ev").fetchone()[0])
    if valid_start >= test_start:
        earlier = con.execute("SELECT quantile_disc(ts_us, 0.889) FROM ev WHERE ts_us < ?", [test_start]).fetchone()[0]
        valid_start = int(earlier) if earlier is not None else test_start
    return valid_start, test_start


def _sample_users(con: duckdb.DuckDBPyConnection, test_start: int, params: dict[str, Any], seed: int) -> None:
    """Create table `sampled` from `ev`: whole users, recent histories, capped test windows."""
    target = params.get("target_events")
    if not target:
        con.execute("CREATE TABLE sampled AS SELECT * FROM ev")
        return
    counts = con.execute(
        """
        SELECT user_id,
               count(*) FILTER (WHERE ts_us < $t) AS n_pre,
               count(*) FILTER (WHERE ts_us >= $t) AS n_test
        FROM ev GROUP BY user_id ORDER BY user_id
        """,
        {"t": test_start},
    ).df()
    rng = np.random.default_rng(seed)
    shuffled = counts.iloc[rng.permutation(len(counts))]
    budget = np.minimum(shuffled["n_pre"], params["max_user_pretest"]) + np.minimum(shuffled["n_test"], params["max_user_test"])
    warm_eval = ((shuffled["n_pre"] > 0) & (shuffled["n_test"] > 0)).to_numpy()
    take_warm = np.cumsum(budget[warm_eval]) <= 0.8 * target
    take_other = np.cumsum(budget[~warm_eval]) <= 0.2 * target
    chosen = pd.concat(
        [shuffled.loc[warm_eval, "user_id"][take_warm.to_numpy()], shuffled.loc[~warm_eval, "user_id"][take_other.to_numpy()]]
    )
    con.register("chosen_df", pd.DataFrame({"user_id": chosen.astype(str).to_numpy()}))
    con.execute(
        f"""
        CREATE TABLE sampled AS
        SELECT user_id, item_id, ts_us, event_seq, session_id, feedback_type, value FROM (
            SELECT ev.*,
                   row_number() OVER (
                       PARTITION BY ev.user_id, ev.ts_us >= {test_start}
                       ORDER BY CASE WHEN ev.ts_us >= {test_start} THEN ev.ts_us ELSE -ev.ts_us END,
                                CASE WHEN ev.ts_us >= {test_start} THEN ev.event_seq ELSE -ev.event_seq END
                   ) AS rn
            FROM ev JOIN chosen_df USING (user_id)
        )
        WHERE (ts_us < {test_start} AND rn <= {int(params['max_user_pretest'])})
           OR (ts_us >= {test_start} AND rn <= {int(params['max_user_test'])})
        """
    )
    max_items = params.get("max_items")
    if max_items:
        con.execute(
            f"""
            CREATE TABLE kept_items AS
            SELECT item_id FROM sampled WHERE ts_us < {test_start}
            GROUP BY item_id ORDER BY count(*) DESC, item_id LIMIT {int(max_items)}
            """
        )
        con.execute("CREATE TABLE sampled2 AS SELECT s.* FROM sampled s JOIN kept_items USING (item_id)")
        con.execute("DROP TABLE sampled")
        con.execute("ALTER TABLE sampled2 RENAME TO sampled")


def materialize(
    clean_dir: Path,
    out_dir: Path,
    *,
    dataset: str,
    tier: str,
    split_rule: str = "quantile",
    test_days: int = 7,
    repeat_policies: tuple[str, ...] = ("exclude_seen",),
    seed: int = 42,
    n_negatives: int = N_NEGATIVES,
    tier_overrides: dict[str, Any] | None = None,
) -> Path:
    if tier not in TIERS:
        raise ValueError(f"Unknown tier {tier}. Known: {sorted(TIERS)}")
    params = {**TIERS[tier], **(tier_overrides or {})}
    clean_dir, out_dir = Path(clean_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear(out_dir)
    con = _con()
    inter = (clean_dir / "interactions.parquet").as_posix()
    items_path = (clean_dir / "items.parquet").as_posix()
    users_path = (clean_dir / "users.parquet").as_posix()

    con.execute(f"CREATE VIEW raw_inter AS SELECT * FROM read_parquet('{inter}', file_row_number=true)")
    con.execute(f"CREATE VIEW raw_items AS SELECT * FROM read_parquet('{items_path}', file_row_number=true)")
    con.execute(f"CREATE VIEW raw_users AS SELECT * FROM read_parquet('{users_path}', file_row_number=true)")
    assert_columns(con, "raw_inter", INTERACTION_COLUMNS)
    assert_columns(con, "raw_items", ITEM_COLUMNS)
    assert_columns(con, "raw_users", USER_COLUMNS)

    con.execute(
        """
        CREATE TABLE ev AS
        SELECT CAST(user_id AS VARCHAR) AS user_id,
               CAST(item_id AS VARCHAR) AS item_id,
               epoch_us(CAST("timestamp" AS TIMESTAMP)) AS ts_us,
               file_row_number AS event_seq,
               COALESCE(CAST(session_id AS VARCHAR), '') AS session_id,
               CAST(feedback_type AS VARCHAR) AS feedback_type,
               CAST(value AS DOUBLE) AS value
        FROM raw_inter
        WHERE "timestamp" IS NOT NULL AND user_id IS NOT NULL AND item_id IS NOT NULL
        """
    )
    if con.execute("SELECT count(*) FROM ev").fetchone()[0] == 0:
        raise SplitError(f"{dataset}: no events in {inter}")
    valid_start, test_start = cutoffs(con, split_rule, test_days)
    _sample_users(con, test_start, params, seed)

    # Sessions: an explicit session_id wins; otherwise a gap of more than 30 minutes starts a new session.
    con.execute(
        f"""
        CREATE TABLE sess AS
        WITH o AS (
            SELECT *, lag(ts_us) OVER (PARTITION BY user_id ORDER BY ts_us, event_seq) AS prev_ts FROM sampled
        ), n AS (
            SELECT *, sum(CASE WHEN prev_ts IS NULL OR ts_us - prev_ts > {SESSION_GAP_US} THEN 1 ELSE 0 END)
                      OVER (PARTITION BY user_id ORDER BY ts_us, event_seq ROWS UNBOUNDED PRECEDING) AS sess_n
            FROM o
        )
        SELECT user_id, item_id, ts_us, event_seq, feedback_type, value,
               CASE WHEN session_id <> '' THEN session_id ELSE user_id || '-' || CAST(sess_n AS VARCHAR) END AS session_id
        FROM n
        """
    )
    con.execute(
        "CREATE TABLE user_map AS SELECT user_id, CAST(row_number() OVER (ORDER BY user_id) AS INTEGER) AS user_idx "
        "FROM (SELECT DISTINCT user_id FROM sess)"
    )
    con.execute(
        "CREATE TABLE item_map AS SELECT item_id, CAST(row_number() OVER (ORDER BY item_id) AS INTEGER) AS item_idx "
        "FROM (SELECT DISTINCT item_id FROM sess)"
    )
    con.execute(
        f"""
        CREATE TABLE indexed AS
        SELECT s.*, um.user_idx, im.item_idx,
               CASE WHEN s.ts_us < {valid_start} THEN 'train' WHEN s.ts_us < {test_start} THEN 'valid' ELSE 'test' END AS part
        FROM sess s JOIN user_map um USING (user_id) JOIN item_map im USING (item_id)
        """
    )
    recent_start = test_start - RECENT_DAYS * DAY_US
    con.execute(
        f"""
        CREATE TABLE item_stats AS
        SELECT item_idx,
               count(*) FILTER (WHERE part <> 'test') AS pretest_count,
               count(*) FILTER (WHERE part <> 'test' AND ts_us >= {recent_start}) AS recent_count,
               min(ts_us) AS first_ts_us
        FROM indexed GROUP BY item_idx
        """
    )
    con.execute(
        "CREATE TABLE user_stats AS SELECT user_idx, count(*) FILTER (WHERE part <> 'test') AS n_pretest, "
        "count(*) FILTER (WHERE part = 'test') AS n_test FROM indexed GROUP BY user_idx"
    )

    # Event files for people (and for RecBole / Recombee exports).
    for part in ("train", "valid", "test"):
        con.execute(
            f"""
            COPY (
              SELECT i.user_id, i.item_id, i.user_idx, i.item_idx, make_timestamp(i.ts_us) AS "timestamp", i.ts_us,
                     i.event_seq, i.session_id, i.feedback_type, i.value,
                     us.n_pretest = 0 AS is_cold_user, COALESCE(st.pretest_count, 0) = 0 AS is_cold_item
              FROM indexed i JOIN user_stats us USING (user_idx) LEFT JOIN item_stats st USING (item_idx)
              WHERE i.part = '{part}' ORDER BY i.ts_us, i.event_seq
            ) TO '{(out_dir / f"{part}.parquet").as_posix()}' (FORMAT PARQUET)
            """
        )

    # Pre-test histories as flat arrays (CSR layout): user u owns items[offsets[u]:offsets[u+1]].
    pre = con.execute(
        "SELECT user_idx, item_idx, ts_us FROM indexed WHERE part <> 'test' ORDER BY user_idx, ts_us, event_seq"
    ).fetchnumpy()
    n_users = int(con.execute("SELECT count(*) FROM user_map").fetchone()[0])
    n_items = int(con.execute("SELECT count(*) FROM item_map").fetchone()[0])
    counts = np.bincount(np.asarray(pre["user_idx"], dtype=np.int64), minlength=n_users + 1)
    offsets = np.zeros(n_users + 2, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    np.save(out_dir / "pretest_offsets.npy", offsets)
    np.save(out_dir / "pretest_items.npy", np.asarray(pre["item_idx"], dtype=np.int32))
    np.save(out_dir / "pretest_ts.npy", np.asarray(pre["ts_us"], dtype=np.int64))

    con.execute(
        """
        CREATE TABLE items_clean AS SELECT * EXCLUDE (rn) FROM (
            SELECT CAST(item_id AS VARCHAR) AS item_id, text, category, image_path,
                   row_number() OVER (PARTITION BY CAST(item_id AS VARCHAR) ORDER BY file_row_number) AS rn
            FROM raw_items) WHERE rn = 1
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT m.item_idx, m.item_id, COALESCE(c.text, '') AS text, COALESCE(c.category, '') AS category,
                 NULLIF(CAST(c.image_path AS VARCHAR), '') AS image_path,
                 COALESCE(s.pretest_count, 0) AS pretest_count, COALESCE(s.recent_count, 0) AS recent_count,
                 s.first_ts_us, COALESCE(s.pretest_count, 0) = 0 AS is_cold
          FROM item_map m LEFT JOIN items_clean c USING (item_id) LEFT JOIN item_stats s USING (item_idx)
          ORDER BY m.item_idx
        ) TO '{(out_dir / "items.parquet").as_posix()}' (FORMAT PARQUET)
        """
    )

    user_frame = con.execute(
        """
        SELECT m.user_idx, m.user_id, COALESCE(c.attributes, '') AS attributes, COALESCE(s.n_pretest, 0) AS n_pretest
        FROM user_map m
        LEFT JOIN (SELECT * EXCLUDE (rn) FROM (
            SELECT CAST(user_id AS VARCHAR) AS user_id, attributes,
                   row_number() OVER (PARTITION BY CAST(user_id AS VARCHAR) ORDER BY file_row_number) AS rn
            FROM raw_users) WHERE rn = 1) c USING (user_id)
        LEFT JOIN user_stats s USING (user_idx)
        ORDER BY m.user_idx
        """
    ).df()
    user_frame["group_label"] = activity_groups(user_frame["n_pretest"].to_numpy())
    user_frame.to_parquet(out_dir / "users.parquet", index=False)

    # Test relevance: first occurrence of each (user, item) in the test window.
    con.execute("CREATE TABLE pre_pairs AS SELECT DISTINCT user_idx, item_idx FROM indexed WHERE part <> 'test'")
    test_items = con.execute(
        """
        WITH firsts AS (
            SELECT user_idx, item_idx, ts_us, event_seq FROM indexed WHERE part = 'test'
            QUALIFY row_number() OVER (PARTITION BY user_idx, item_idx ORDER BY ts_us, event_seq) = 1
        )
        SELECT f.user_idx, f.item_idx, f.ts_us AS first_ts_us,
               CAST(row_number() OVER (PARTITION BY f.user_idx ORDER BY f.ts_us, f.event_seq) AS INTEGER) AS test_rank,
               pp.user_idx IS NOT NULL AS is_repeat,
               COALESCE(st.pretest_count, 0) = 0 AS is_cold_item
        FROM firsts f
        LEFT JOIN pre_pairs pp ON pp.user_idx = f.user_idx AND pp.item_idx = f.item_idx
        LEFT JOIN item_stats st ON st.item_idx = f.item_idx
        ORDER BY f.user_idx, test_rank
        """
    ).df()
    test_items.to_parquet(out_dir / "test_items.parquet", index=False)

    n_pre = user_frame.set_index("user_idx")["n_pretest"]
    eval_users = _eval_users(test_items, n_pre, repeat_policies, params, seed)
    eval_users.to_parquet(out_dir / "eval_users.parquet", index=False)
    n_eval_warm = int(eval_users["is_warm"].sum())

    item_pop = np.zeros(n_items + 1, dtype=np.int64)
    stats = con.execute("SELECT item_idx, pretest_count FROM item_stats").fetchnumpy()
    item_pop[np.asarray(stats["item_idx"], dtype=np.int64)] = np.asarray(stats["pretest_count"], dtype=np.int64)
    _write_candidates(out_dir, eval_users, offsets, np.load(out_dir / "pretest_items.npy"), item_pop, n_negatives, seed)

    totals = con.execute(
        "SELECT count(*) FILTER (WHERE part='train'), count(*) FILTER (WHERE part='valid'), count(*) FILTER (WHERE part='test') FROM indexed"
    ).fetchone()
    test_users = test_items.groupby("user_idx").size()
    warm_test_users = int((n_pre.reindex(test_users.index).fillna(0) > 0).sum())
    clean_stat = (clean_dir / "interactions.parquet").stat()
    identity = {
        "dataset": dataset,
        "tier": tier,
        "split_rule": split_rule,
        "test_days": test_days,
        "valid_start_us": valid_start,
        "test_start_us": test_start,
        "params": params,
        "seed": seed,
        "n_negatives": n_negatives,
        "repeat_policies": list(repeat_policies),
        "clean_size": clean_stat.st_size,
        "n_users": n_users,
        "n_items": n_items,
        "contract": CONTRACT_VERSION,
    }
    split_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    meta = {
        "dataset": dataset,
        "tier": tier,
        "contract_version": CONTRACT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "split_hash": split_hash,
        "split_rule": split_rule,
        "valid_start": _iso(valid_start),
        "test_start": _iso(test_start),
        "valid_start_us": valid_start,
        "test_start_us": test_start,
        "n_users": n_users,
        "n_items": n_items,
        "n_pretest": int(totals[0] + totals[1]),
        "n_train": int(totals[0]),
        "n_valid": int(totals[1]),
        "n_test": int(totals[2]),
        "n_test_users": int(len(test_users)),
        "n_warm_test_users": warm_test_users,
        "n_cold_test_users": int(len(test_users) - warm_test_users),
        "n_eval_warm": n_eval_warm,
        "n_eval_cold": int((~eval_users["is_warm"]).sum()),
        "n_cold_items": int((item_pop[1:] == 0).sum()),
        "repeat_share": float(test_items["is_repeat"].mean()) if len(test_items) else 0.0,
        "repeat_policies": list(repeat_policies),
        "primary_policy": repeat_policies[0],
        "n_negatives": n_negatives,
        "seed": seed,
        "tier_params": params,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    if n_eval_warm < int(params["min_eval_users"]):
        raise SplitError(
            f"{dataset}/{tier}: only {n_eval_warm} warm eval users (minimum {params['min_eval_users']}). "
            "The split was written for inspection but must not be benchmarked."
        )
    return out_dir


def activity_groups(n_pretest: np.ndarray) -> list[str]:
    """cold for users without pre-test events; light/medium/heavy terciles for the rest."""
    n_pretest = np.asarray(n_pretest)
    warm = n_pretest[n_pretest > 0]
    if len(warm) == 0:
        return ["cold"] * len(n_pretest)
    low, high = np.quantile(warm, [1 / 3, 2 / 3])
    labels = []
    for count in n_pretest:
        if count == 0:
            labels.append("cold")
        elif count <= low:
            labels.append("light")
        elif count <= high:
            labels.append("medium")
        else:
            labels.append("heavy")
    return labels


def _eval_users(
    test_items: pd.DataFrame,
    n_pre: pd.Series,
    repeat_policies: tuple[str, ...],
    params: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    """Sample eval users and record the next-item target under each repeat policy."""
    primary = repeat_policies[0]
    frame = test_items.copy()
    frame["warm"] = n_pre.reindex(frame["user_idx"]).fillna(0).to_numpy() > 0
    relevant = frame if primary == "allow_repeats" else frame[~frame["is_repeat"]]
    warm_candidates = np.sort(relevant.loc[relevant["warm"], "user_idx"].unique())
    cold_candidates = np.sort(frame.loc[~frame["warm"], "user_idx"].unique())
    rng = np.random.default_rng(seed)
    cap = int(params["max_eval_users"])
    warm = np.sort(rng.permutation(warm_candidates)[:cap]) if len(warm_candidates) > cap else warm_candidates
    cold = np.sort(rng.permutation(cold_candidates)[:cap]) if len(cold_candidates) > cap else cold_candidates
    out = pd.DataFrame(
        {"user_idx": np.concatenate([warm, cold]).astype(np.int32), "is_warm": [True] * len(warm) + [False] * len(cold)}
    )
    for policy in repeat_policies:
        pool = frame if policy == "allow_repeats" else frame[~frame["is_repeat"]]
        first = pool.sort_values(["user_idx", "test_rank"]).drop_duplicates("user_idx").set_index("user_idx")["item_idx"]
        column = "next_item" if policy == primary else f"next_item__{policy}"
        out[column] = first.reindex(out["user_idx"]).fillna(0).astype(np.int32).to_numpy()
    return out


def _write_candidates(
    out_dir: Path,
    eval_users: pd.DataFrame,
    offsets: np.ndarray,
    pre_items: np.ndarray,
    item_pop: np.ndarray,
    n_negatives: int,
    seed: int,
) -> None:
    """Secondary protocol: the next item plus n_negatives unseen warm items per warm eval user."""
    warm_items = np.flatnonzero(item_pop > 0)
    warm_items = warm_items[warm_items > 0]
    rng = np.random.default_rng(seed + 1)
    rows_u, rows_i, rows_y = [], [], []
    users = eval_users.loc[eval_users["is_warm"] & (eval_users["next_item"] > 0)]
    for user, positive in zip(users["user_idx"].to_numpy(), users["next_item"].to_numpy()):
        seen = set(pre_items[offsets[user] : offsets[user + 1]].tolist())
        seen.add(int(positive))
        chosen: list[int] = []
        picked: set[int] = set()
        # Rejection sampling; a user who has seen almost every item simply gets fewer negatives.
        for _ in range(50):
            if len(chosen) >= n_negatives:
                break
            draw = rng.choice(warm_items, size=max(2 * (n_negatives - len(chosen)), 8))
            for item in draw.tolist():
                if item in seen or item in picked:
                    continue
                picked.add(item)
                chosen.append(item)
                if len(chosen) >= n_negatives:
                    break
        rows_u.extend([int(user)] * (1 + len(chosen)))
        rows_i.extend([int(positive), *chosen])
        rows_y.extend([1] + [0] * len(chosen))
    table = pa.table(
        {
            "user_idx": pa.array(rows_u, type=pa.int32()),
            "item_idx": pa.array(rows_i, type=pa.int32()),
            "label": pa.array(rows_y, type=pa.int8()),
        }
    )
    pq.write_table(table, out_dir / "candidates.parquet")
