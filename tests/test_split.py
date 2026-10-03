"""The split: time cutoffs, ordering, relevance sets, eval users, candidates."""

from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from recbench.data import SplitError, TrainView
from recbench.pipeline.materialize import cutoffs, materialize
from recbench.pipeline.toy import write_toy_clean


def test_cutoffs_do_not_depend_on_the_time_zone(tmp_path):
    write_toy_clean(tmp_path / "clean")
    results = []
    for zone in ("UTC", "Asia/Ho_Chi_Minh", "America/Los_Angeles"):
        con = duckdb.connect()
        con.execute(f"SET TimeZone='{zone}'")
        con.execute(
            f"CREATE TABLE ev AS SELECT epoch_us(CAST(\"timestamp\" AS TIMESTAMP)) AS ts_us "
            f"FROM read_parquet('{(tmp_path / 'clean' / 'interactions.parquet').as_posix()}')"
        )
        results.append(cutoffs(con, "quantile", 7) + cutoffs(con, "last_days", 7))
    assert results[0] == results[1] == results[2]


def test_events_respect_the_cutoff(toy_split):
    meta = json.loads((toy_split / "meta.json").read_text())
    pre = np.load(toy_split / "pretest_ts.npy")
    test = pq.read_table(toy_split / "test.parquet").to_pandas()
    assert pre.max() < meta["test_start_us"] <= test["ts_us"].min()
    valid = pq.read_table(toy_split / "valid.parquet").to_pandas()
    assert valid["ts_us"].min() >= meta["valid_start_us"]


def test_rebuilding_gives_identical_files(tmp_path):
    write_toy_clean(tmp_path / "clean")
    a = materialize(tmp_path / "clean", tmp_path / "a", dataset="toy", tier="full", tier_overrides={"min_eval_users": 1})
    b = materialize(tmp_path / "clean", tmp_path / "b", dataset="toy", tier="full", tier_overrides={"min_eval_users": 1})
    for name in ("pretest_items.npy", "pretest_offsets.npy"):
        assert np.array_equal(np.load(a / name), np.load(b / name))
    pd.testing.assert_frame_equal(pd.read_parquet(a / "test_items.parquet"), pd.read_parquet(b / "test_items.parquet"))
    assert json.loads((a / "meta.json").read_text())["split_hash"] == json.loads((b / "meta.json").read_text())["split_hash"]


def test_ties_are_broken_by_file_order(tmp_path):
    clean = tmp_path / "clean"
    clean.mkdir()
    same_day = pd.Timestamp("2020-01-01")
    rows = [("u1", f"i{i}", same_day) for i in (3, 1, 2)] + [("u1", "i9", pd.Timestamp("2020-02-01"))]
    rows += [(f"u{j}", f"i{j % 5}", pd.Timestamp("2020-01-01") + pd.Timedelta(days=j)) for j in range(2, 40)]
    pd.DataFrame({"user_id": [r[0] for r in rows], "item_id": [r[1] for r in rows], "timestamp": [r[2] for r in rows],
                  "session_id": "", "feedback_type": "implicit", "value": 1.0}).to_parquet(clean / "interactions.parquet")
    pd.DataFrame({"item_id": [f"i{i}" for i in range(10)], "text": "", "category": "", "image_path": None}).to_parquet(clean / "items.parquet")
    pd.DataFrame({"user_id": ["u1"], "attributes": "", "group_label": None}).to_parquet(clean / "users.parquet")
    out = materialize(clean, tmp_path / "split", dataset="ties", tier="full", tier_overrides={"min_eval_users": 0})
    view = TrainView(out)
    u1 = int(np.flatnonzero(view.user_ids == "u1")[0])
    assert [view.item_ids[i] for i in view.user_items(u1)][:3] == ["i3", "i1", "i2"]


def test_eval_users_have_history_and_a_new_relevant_item(toy_split):
    view = TrainView(toy_split)
    eval_users = pd.read_parquet(toy_split / "eval_users.parquet")
    test_items = pd.read_parquet(toy_split / "test_items.parquet")
    warm = eval_users[eval_users["is_warm"]]
    assert (view.user_lengths[warm["user_idx"]] > 0).all()
    new_items = test_items[~test_items["is_repeat"]]
    for user, nxt in zip(warm["user_idx"], warm["next_item"]):
        mine = new_items[new_items["user_idx"] == user].sort_values("test_rank")
        assert nxt == mine["item_idx"].iloc[0]
        assert nxt not in set(view.user_items(int(user)))


def test_candidates_are_positive_plus_unseen_warm_items(toy_split):
    view = TrainView(toy_split)
    cands = pd.read_parquet(toy_split / "candidates.parquet")
    for user, group in cands.groupby("user_idx"):
        assert group["label"].sum() == 1
        negatives = group.loc[group["label"] == 0, "item_idx"]
        assert not set(negatives) & set(view.user_items(int(user)))
        assert (view.item_pop[negatives.to_numpy()] > 0).all()


def test_too_few_eval_users_fails_loudly(tmp_path):
    write_toy_clean(tmp_path / "clean", n_users=8)
    with pytest.raises(SplitError):
        materialize(tmp_path / "clean", tmp_path / "split", dataset="toy", tier="full", tier_overrides={"min_eval_users": 50})


def test_history_batch_is_right_aligned(toy):
    view, _ = toy
    users = view.warm_users()[:5]
    batch = view.history_batch(users, 6)
    for row, user in enumerate(users):
        items = view.user_items(int(user))[-6:]
        assert list(batch.items[row, -len(items):]) == list(items)
        assert batch.lengths[row] == len(items)
        left = batch.left_aligned()[row]
        assert list(left[: len(items)]) == list(items) and not left[len(items):].any()


@pytest.mark.parametrize("tier,overrides", [("full", {}), ("smoke", {"target_events": 1500, "max_user_pretest": 20})])
def test_validation_fold_has_no_real_test_events_and_keeps_the_users(tmp_path, tier, overrides):
    write_toy_clean(tmp_path / "clean")
    settings = {"min_eval_users": 1, **overrides}
    base = materialize(tmp_path / "clean", tmp_path / "base", dataset="toy", tier=tier, tier_overrides=settings)
    fold = materialize(tmp_path / "clean", tmp_path / "fold", dataset="toy", tier=tier, tier_overrides=settings, fold="valid")
    bm, fm = (json.loads((p / "meta.json").read_text()) for p in (base, fold))
    assert fm["tier"] == f"{tier}-val" and fm["split_hash"] != bm["split_hash"]
    assert fm["test_start_us"] == bm["valid_start_us"]  # the fold's test window is the real validation window
    for part in ("train", "valid", "test"):
        assert (pd.read_parquet(fold / f"{part}.parquet")["ts_us"] < bm["test_start_us"]).all()
    assert pd.read_parquet(fold / "test.parquet")["ts_us"].min() >= bm["valid_start_us"]
    assert np.load(fold / "pretest_ts.npy").max() < bm["valid_start_us"]
    base_users = set(pd.read_parquet(base / "users.parquet")["user_id"])
    assert set(pd.read_parquet(fold / "users.parquet")["user_id"]) <= base_users  # same sampled users, never new ones
    TrainView(fold)  # a fold is an ordinary split for models and the evaluator
