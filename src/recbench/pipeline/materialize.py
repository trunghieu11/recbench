"""Turn clean interaction tables into a temporal split, streamed sequences, and one candidate file."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from recbench.protocol import N_NEGATIVES
from recbench.schema import (
    INTERACTION_COLUMNS,
    ITEM_COLUMNS,
    USER_COLUMNS,
    assert_columns,
)
from recbench.store import SplitStore

TIER_CAPS = {"smoke": 50_000, "standard": 1_000_000, "full": None}


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    return con


def _write_clean(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    users: pd.DataFrame,
    clean_dir: Path,
) -> None:
    clean_dir.mkdir(parents=True, exist_ok=True)
    for frame, cols in (
        (interactions, INTERACTION_COLUMNS),
        (items, ITEM_COLUMNS),
        (users, USER_COLUMNS),
    ):
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError(f"Clean table missing {missing}")
    interactions = interactions.copy()
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], utc=True).dt.tz_localize(None)
    interactions.to_parquet(clean_dir / "interactions.parquet", index=False)
    items.to_parquet(clean_dir / "items.parquet", index=False)
    users.to_parquet(clean_dir / "users.parquet", index=False)


def materialize(
    clean_dir: Path,
    out_dir: Path,
    *,
    dataset: str,
    tier: str,
    split_mode: str = "quantile",
    seed: int = 42,
    n_negatives: int = N_NEGATIVES,
) -> SplitStore:
    """Build train/valid/test, sequences, and the shared candidate file.

    Histories contain only events strictly before the test cutoff.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    con = _con()
    inter = (clean_dir / "interactions.parquet").as_posix()
    items = (clean_dir / "items.parquet").as_posix()
    users = (clean_dir / "users.parquet").as_posix()
    con.execute(f"CREATE TABLE interactions AS SELECT * FROM read_parquet('{inter}')")
    con.execute(f"CREATE TABLE items_raw AS SELECT * FROM read_parquet('{items}')")
    con.execute(f"CREATE TABLE users_raw AS SELECT * FROM read_parquet('{users}')")
    assert_columns(con, "interactions", INTERACTION_COLUMNS)
    assert_columns(con, "items_raw", ITEM_COLUMNS)
    assert_columns(con, "users_raw", USER_COLUMNS)

    cap = TIER_CAPS.get(tier)
    if cap is None or tier == "full":
        con.execute("CREATE TABLE capped AS SELECT * FROM interactions")
    else:
        con.execute(
            f"""
            CREATE TABLE capped AS
            SELECT * FROM interactions
            ORDER BY timestamp DESC
            LIMIT {int(cap)}
            """
        )

    con.execute(
        """
        CREATE TABLE sessionized AS
        WITH ordered AS (
            SELECT *,
                   CASE
                     WHEN session_id IS NOT NULL AND CAST(session_id AS VARCHAR) <> '' THEN 0
                     WHEN lag(timestamp) OVER (PARTITION BY user_id ORDER BY timestamp) IS NULL THEN 1
                     WHEN date_diff(
                            'minute',
                            lag(timestamp) OVER (PARTITION BY user_id ORDER BY timestamp),
                            timestamp
                          ) > 30 THEN 1
                     ELSE 0
                   END AS new_sess
            FROM capped
        ), numbered AS (
            SELECT *,
                   SUM(new_sess) OVER (PARTITION BY user_id ORDER BY timestamp) AS sess_n
            FROM ordered
        )
        SELECT user_id, item_id, timestamp, feedback_type, value,
               CASE
                 WHEN session_id IS NOT NULL AND CAST(session_id AS VARCHAR) <> '' THEN CAST(session_id AS VARCHAR)
                 ELSE user_id || '-' || CAST(sess_n AS VARCHAR)
               END AS session_id
        FROM numbered
        """
    )

    if split_mode == "hm_last_7_days" and tier == "full":
        con.execute(
            """
            CREATE TABLE bounds AS
            SELECT max(timestamp) - INTERVAL 7 DAY AS test_start FROM sessionized
            """
        )
    else:
        con.execute(
            """
            CREATE TABLE bounds AS
            SELECT quantile_cont(epoch(timestamp), 0.9) AS test_epoch FROM sessionized
            """
        )
        # epoch quantile returns a double; convert back
        test_epoch = con.execute("SELECT test_epoch FROM bounds").fetchone()[0]
        con.execute("DROP TABLE bounds")
        con.execute(
            "CREATE TABLE bounds AS SELECT to_timestamp(?)::TIMESTAMP AS test_start",
            [float(test_epoch)],
        )

    test_start = con.execute("SELECT test_start FROM bounds").fetchone()[0]
    # Valid is the slice of pre-cutoff time from the 80th to the 90th percentile
    # of the whole stream when using quantiles. For the H&M week, valid is the
    # 10% of pre-cutoff events closest to the cutoff.
    if split_mode == "hm_last_7_days" and tier == "full":
        valid_epoch = con.execute(
            """
            SELECT quantile_cont(epoch(timestamp), 0.9)
            FROM sessionized WHERE timestamp < ?
            """,
            [test_start],
        ).fetchone()[0]
        valid_start = con.execute("SELECT to_timestamp(?)::TIMESTAMP", [float(valid_epoch)]).fetchone()[0]
    else:
        valid_epoch = con.execute(
            "SELECT quantile_cont(epoch(timestamp), 0.8) FROM sessionized"
        ).fetchone()[0]
        valid_start = con.execute("SELECT to_timestamp(?)::TIMESTAMP", [float(valid_epoch)]).fetchone()[0]

    con.execute(
        """
        CREATE TABLE first_user AS
        SELECT user_id, min(timestamp) AS first_ts FROM sessionized GROUP BY 1
        """
    )
    con.execute(
        """
        CREATE TABLE first_item AS
        SELECT item_id, min(timestamp) AS first_ts FROM sessionized GROUP BY 1
        """
    )
    con.execute(
        """
        CREATE TABLE labeled AS
        SELECT s.*,
               CASE
                 WHEN s.timestamp < ? THEN 'train'
                 WHEN s.timestamp < ? THEN 'valid'
                 ELSE 'test'
               END AS part,
               (fu.first_ts >= ?) AS is_cold_user,
               (fi.first_ts >= ?) AS is_cold_item
        FROM sessionized s
        JOIN first_user fu USING (user_id)
        JOIN first_item fi USING (item_id)
        """,
        [valid_start, test_start, test_start, test_start],
    )

    con.execute(
        """
        CREATE TABLE user_map AS
        SELECT user_id, CAST(row_number() OVER (ORDER BY user_id) AS INTEGER) AS user_idx
        FROM (SELECT DISTINCT user_id FROM labeled)
        """
    )
    con.execute(
        """
        CREATE TABLE item_map AS
        SELECT item_id, CAST(row_number() OVER (ORDER BY item_id) AS INTEGER) AS item_idx
        FROM (SELECT DISTINCT item_id FROM labeled)
        """
    )
    con.execute(
        """
        CREATE TABLE indexed AS
        SELECT l.*, um.user_idx, im.item_idx
        FROM labeled l
        JOIN user_map um USING (user_id)
        JOIN item_map im USING (item_id)
        """
    )

    for part in ("train", "valid", "test"):
        con.execute(
            f"""
            COPY (
              SELECT user_id, item_id, user_idx, item_idx, timestamp, session_id,
                     feedback_type, value, is_cold_user, is_cold_item
              FROM indexed WHERE part = '{part}'
              ORDER BY timestamp
            ) TO '{(out_dir / f"{part}.parquet").as_posix()}' (FORMAT PARQUET)
            """
        )

    con.execute(
        f"""
        COPY (
          SELECT m.item_idx, m.item_id,
                 COALESCE(i.text, '') AS text,
                 COALESCE(i.category, '') AS category,
                 i.image_path
          FROM item_map m
          LEFT JOIN items_raw i USING (item_id)
          ORDER BY m.item_idx
        ) TO '{(out_dir / "items.parquet").as_posix()}' (FORMAT PARQUET)
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT m.user_idx, m.user_id,
                 COALESCE(u.attributes, '') AS attributes,
                 u.group_label
          FROM user_map m
          LEFT JOIN users_raw u USING (user_id)
          ORDER BY m.user_idx
        ) TO '{(out_dir / "users.parquet").as_posix()}' (FORMAT PARQUET)
        """
    )

    # Sequences: aggregate in DuckDB, then stream row groups into the memmap-friendly parquet.
    con.execute(
        f"""
        COPY (
          SELECT user_idx,
                 list(item_idx ORDER BY timestamp) AS history
          FROM indexed
          WHERE timestamp < ?
          GROUP BY user_idx
        ) TO '{(out_dir / "_train_hist.parquet").as_posix()}' (FORMAT PARQUET)
        """,
        [test_start],
    )
    con.execute(
        f"""
        COPY (
          SELECT user_idx,
                 item_idx AS target_item,
                 is_cold_user
          FROM (
            SELECT *, row_number() OVER (PARTITION BY user_idx ORDER BY timestamp) AS rn
            FROM indexed WHERE part = 'test'
          ) t
          WHERE rn = 1
        ) TO '{(out_dir / "targets.parquet").as_posix()}' (FORMAT PARQUET)
        """
    )
    _write_sequences(out_dir)

    n_users = con.execute("SELECT count(*) FROM user_map").fetchone()[0]
    n_items = con.execute("SELECT count(*) FROM item_map").fetchone()[0]
    n_train = con.execute("SELECT count(*) FROM indexed WHERE part = 'train'").fetchone()[0]
    n_test = con.execute("SELECT count(*) FROM indexed WHERE part = 'test'").fetchone()[0]
    _write_candidates(out_dir, n_items=int(n_items), n_negatives=n_negatives, seed=seed)

    meta = {
        "dataset": dataset,
        "tier": tier,
        "split_mode": split_mode if not (split_mode == "hm_last_7_days" and tier != "full") else "quantile",
        "requested_split_mode": split_mode,
        "test_start": str(test_start),
        "valid_start": str(valid_start),
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_negatives": int(n_negatives),
        "contract_version": "1",
        "seed": seed,
        "protocol": "temporal cutoff; histories stop at test_start; shared candidate file",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return SplitStore(out_dir, dataset, tier)


def _write_sequences(out_dir: Path) -> None:
    """Stream training histories and attach the first test target. Keep the last 200 items."""
    hist_file = pq.ParquetFile(out_dir / "_train_hist.parquet")
    targets = pq.read_table(out_dir / "targets.parquet").to_pandas()
    target_of = {
        int(u): (int(t), bool(c))
        for u, t, c in zip(targets["user_idx"], targets["target_item"], targets["is_cold_user"])
    }
    written: set[int] = set()
    schema = pa.schema(
        [
            ("user_idx", pa.int32()),
            ("target_item", pa.int32()),
            ("history", pa.list_(pa.int32())),
            ("is_cold_user", pa.bool_()),
        ]
    )
    writer = pq.ParquetWriter(out_dir / "sequences.parquet", schema)
    try:
        for batch in hist_file.iter_batches(batch_size=4096):
            users = batch.column("user_idx").to_pylist()
            histories = batch.column("history").to_pylist()
            keep_u, keep_t, keep_h, keep_c = [], [], [], []
            for user, hist in zip(users, histories):
                if int(user) not in target_of:
                    continue
                target, cold = target_of[int(user)]
                tail = [int(x) for x in (hist or [])[-200:]]
                keep_u.append(int(user))
                keep_t.append(target)
                keep_h.append(tail)
                keep_c.append(cold)
                written.add(int(user))
            if not keep_u:
                continue
            writer.write_table(
                pa.table(
                    {
                        "user_idx": pa.array(keep_u, type=pa.int32()),
                        "target_item": pa.array(keep_t, type=pa.int32()),
                        "history": pa.array(keep_h, type=pa.list_(pa.int32())),
                        "is_cold_user": pa.array(keep_c, type=pa.bool_()),
                    }
                )
            )
        missing_u, missing_t, missing_h, missing_c = [], [], [], []
        for user, (target, cold) in target_of.items():
            if user in written:
                continue
            missing_u.append(user)
            missing_t.append(target)
            missing_h.append([])
            missing_c.append(cold)
        if missing_u:
            writer.write_table(
                pa.table(
                    {
                        "user_idx": pa.array(missing_u, type=pa.int32()),
                        "target_item": pa.array(missing_t, type=pa.int32()),
                        "history": pa.array(missing_h, type=pa.list_(pa.int32())),
                        "is_cold_user": pa.array(missing_c, type=pa.bool_()),
                    }
                )
            )
    finally:
        writer.close()


def _write_candidates(out_dir: Path, *, n_items: int, n_negatives: int, seed: int) -> None:
    targets = pq.read_table(out_dir / "targets.parquet").to_pandas()
    if targets.empty or n_items < 2:
        empty = pd.DataFrame(
            columns=["user_idx", "item_idx", "label", "is_cold_user", "is_cold_item"]
        )
        empty.to_parquet(out_dir / "candidates.parquet", index=False)
        return
    rng = np.random.default_rng(seed)
    catalog = np.arange(1, n_items + 1, dtype=np.int32)
    n_neg = min(n_negatives, max(n_items - 1, 1))
    # Seen items are excluded when the train set for that user is small enough to load per chunk.
    train = pq.read_table(out_dir / "train.parquet", columns=["user_idx", "item_idx"]).to_pandas()
    seen: dict[int, set[int]] = {}
    for user, item in zip(train["user_idx"].to_numpy(), train["item_idx"].to_numpy()):
        seen.setdefault(int(user), set()).add(int(item))

    cold_items = set()
    if (out_dir / "test.parquet").exists():
        test = pq.read_table(out_dir / "test.parquet", columns=["item_idx", "is_cold_item"]).to_pandas()
        cold_items = set(int(i) for i, flag in zip(test["item_idx"], test["is_cold_item"]) if bool(flag))

    schema = pa.schema(
        [
            ("user_idx", pa.int32()),
            ("item_idx", pa.int32()),
            ("label", pa.int8()),
            ("is_cold_user", pa.bool_()),
            ("is_cold_item", pa.bool_()),
        ]
    )
    writer = pq.ParquetWriter(out_dir / "candidates.parquet", schema)
    users = targets["user_idx"].to_numpy()
    positives = targets["target_item"].to_numpy()
    cold_users = targets["is_cold_user"].to_numpy()
    try:
        for start in range(0, len(users), 2000):
            rows_u, rows_i, rows_y, rows_cu, rows_ci = [], [], [], [], []
            stop = min(start + 2000, len(users))
            for user, pos, cold_u in zip(users[start:stop], positives[start:stop], cold_users[start:stop]):
                user = int(user)
                pos = int(pos)
                blocked = seen.get(user, set()) | {pos}
                pool = catalog if len(blocked) > n_items * 0.5 else None
                chosen: list[int] = []
                guard = 0
                while len(chosen) < n_neg and guard < n_neg * 20:
                    guard += 1
                    draw = int(rng.choice(catalog if pool is None else pool))
                    if draw in blocked or draw in chosen:
                        continue
                    chosen.append(draw)
                if len(chosen) < n_neg:
                    for draw in catalog:
                        if int(draw) in blocked or int(draw) in chosen:
                            continue
                        chosen.append(int(draw))
                        if len(chosen) >= n_neg:
                            break
                rows_u.append(user)
                rows_i.append(pos)
                rows_y.append(1)
                rows_cu.append(bool(cold_u))
                rows_ci.append(pos in cold_items)
                for item in chosen:
                    rows_u.append(user)
                    rows_i.append(item)
                    rows_y.append(0)
                    rows_cu.append(bool(cold_u))
                    rows_ci.append(item in cold_items)
            writer.write_table(
                pa.table(
                    {
                        "user_idx": pa.array(rows_u, type=pa.int32()),
                        "item_idx": pa.array(rows_i, type=pa.int32()),
                        "label": pa.array(rows_y, type=pa.int8()),
                        "is_cold_user": pa.array(rows_cu, type=pa.bool_()),
                        "is_cold_item": pa.array(rows_ci, type=pa.bool_()),
                    }
                )
            )
    finally:
        writer.close()
