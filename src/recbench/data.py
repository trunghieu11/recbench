"""What a model is allowed to see.

TrainView exposes only events strictly before the test cutoff ("pre-test" =
train + valid) plus item/user metadata. It has no path to test events, eval
users, or the sampled candidate file, so a method cannot leak test labels into
training. The evaluator reads those files separately (recbench.evaluation).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

DAY_US = 86_400_000_000
RECENT_DAYS = 28

# meta.json keys a model may read. Test-side counts stay with the evaluator.
TRAIN_META_KEYS = (
    "dataset",
    "tier",
    "contract_version",
    "protocol_version",
    "split_hash",
    "n_users",
    "n_items",
    "n_pretest",
    "test_start",
    "test_start_us",
    "valid_start_us",
    "seed",
)


class SplitError(Exception):
    """A split is missing, stale, or too small to evaluate."""


@dataclass
class HistoryBatch:
    """Pre-test item histories for a batch of users.

    items is right-aligned: the most recent item sits in the last column and
    padding (item 0) fills the left. lengths counts the real items per row.
    """

    items: np.ndarray
    lengths: np.ndarray
    times: np.ndarray | None = None  # event times in microseconds, aligned with items (0 for padding)

    @property
    def seq_len(self) -> int:
        return int(self.items.shape[1])

    def left_aligned(self) -> np.ndarray:
        """Same histories with items first and padding at the end (RecBole's layout)."""
        batch, width = self.items.shape
        cols = np.arange(width)[None, :]
        source = cols + (width - self.lengths)[:, None]
        valid = cols < self.lengths[:, None]
        gathered = np.take_along_axis(self.items, np.clip(source, 0, width - 1), axis=1)
        return np.where(valid, gathered, 0)

    def last_items(self) -> np.ndarray:
        return np.where(self.lengths > 0, self.items[:, -1], 0)


class TrainView:
    """Read-only access to the pre-test part of one split."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        meta_path = self.root / "meta.json"
        if not meta_path.exists():
            raise SplitError(f"No split at {self.root}")
        full_meta = json.loads(meta_path.read_text())
        if str(full_meta.get("contract_version")) != "2":
            raise SplitError(f"{self.root} uses contract v{full_meta.get('contract_version')}; rebuild it with prepare")
        self.meta = {key: full_meta[key] for key in TRAIN_META_KEYS if key in full_meta}
        self.dataset: str = full_meta["dataset"]
        self.tier: str = full_meta["tier"]
        self.split_hash: str = full_meta["split_hash"]
        self.n_users: int = int(full_meta["n_users"])
        self.n_items: int = int(full_meta["n_items"])
        self._offsets = np.load(self.root / "pretest_offsets.npy", mmap_mode="r")
        self._items = np.load(self.root / "pretest_items.npy", mmap_mode="r")
        self._ts = np.load(self.root / "pretest_ts.npy", mmap_mode="r")
        items = pq.read_table(self.root / "items.parquet").to_pandas().sort_values("item_idx")
        users = pq.read_table(self.root / "users.parquet").to_pandas().sort_values("user_idx")
        self.item_ids = np.concatenate([[""], items["item_id"].astype(str).to_numpy()]).astype(object)
        self.user_ids = np.concatenate([[""], users["user_id"].astype(str).to_numpy()]).astype(object)
        self.item_text = np.concatenate([[""], items["text"].fillna("").astype(str).to_numpy()]).astype(object)
        self.item_category = np.concatenate([[""], items["category"].fillna("").astype(str).to_numpy()]).astype(object)
        image = items["image_path"].astype(object).where(items["image_path"].notna(), "")
        self.item_image_path = np.concatenate([[""], image.astype(str).to_numpy()]).astype(object)
        self.item_pop = np.concatenate([[0], items["pretest_count"].to_numpy()]).astype(np.int64)
        self.item_recent_pop = np.concatenate([[0], items["recent_count"].to_numpy()]).astype(np.int64)
        self.user_attributes = np.concatenate([[""], users["attributes"].fillna("").astype(str).to_numpy()]).astype(object)

    # ----- per-user histories -----
    @cached_property
    def user_lengths(self) -> np.ndarray:
        return np.diff(np.asarray(self._offsets)).astype(np.int64)

    def user_items(self, user: int) -> np.ndarray:
        return np.asarray(self._items[self._offsets[user] : self._offsets[user + 1]])

    def user_times(self, user: int) -> np.ndarray:
        return np.asarray(self._ts[self._offsets[user] : self._offsets[user + 1]])

    def history_batch(self, users: np.ndarray, seq_len: int) -> HistoryBatch:
        users = np.asarray(users, dtype=np.int64)
        starts = np.asarray(self._offsets)[users]
        ends = np.asarray(self._offsets)[users + 1]
        lengths = np.minimum(ends - starts, seq_len)
        cols = np.arange(seq_len)[None, :]
        pad = seq_len - lengths
        index = ends[:, None] - seq_len + cols
        valid = cols >= pad[:, None]
        flat = np.asarray(self._items)
        stamps = np.asarray(self._ts)
        safe = np.clip(index, 0, max(len(flat) - 1, 0))
        items = np.where(valid, flat[safe], 0) if len(flat) else np.zeros((len(users), seq_len), dtype=np.int64)
        times = np.where(valid, stamps[safe], 0) if len(flat) else np.zeros((len(users), seq_len), dtype=np.int64)
        return HistoryBatch(items=items.astype(np.int64), lengths=lengths.astype(np.int64), times=times.astype(np.int64))

    # ----- matrices and frames -----
    @cached_property
    def interaction_counts(self) -> sp.csr_matrix:
        """User x item counts of pre-test events, shape [n_users + 1, n_items + 1]."""
        lengths = self.user_lengths
        rows = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
        cols = np.asarray(self._items, dtype=np.int64)
        data = np.ones(len(cols), dtype=np.float32)
        matrix = sp.csr_matrix((data, (rows, cols)), shape=(self.n_users + 1, self.n_items + 1))
        matrix.sum_duplicates()
        return matrix

    @cached_property
    def seen(self) -> sp.csr_matrix:
        """Binary user x item matrix: 1 where the user interacted before the cutoff."""
        binary = self.interaction_counts.copy()
        binary.data[:] = 1.0
        return binary

    def events(self) -> pd.DataFrame:
        """Pre-test events, sorted by user then time: user_idx, item_idx, ts_us."""
        lengths = self.user_lengths
        return pd.DataFrame(
            {
                "user_idx": np.repeat(np.arange(len(lengths), dtype=np.int64), lengths),
                "item_idx": np.asarray(self._items, dtype=np.int64),
                "ts_us": np.asarray(self._ts, dtype=np.int64),
            }
        )

    def export_frame(self) -> pd.DataFrame:
        """Pre-test events with raw ids, feedback, and values (for RecBole and Recombee uploads)."""
        cols = ["user_id", "item_id", "user_idx", "item_idx", "ts_us", "event_seq", "feedback_type", "value"]
        parts = [pq.read_table(self.root / f"{part}.parquet", columns=cols).to_pandas() for part in ("train", "valid")]
        frame = pd.concat(parts, ignore_index=True)
        frame = frame.sort_values(["user_idx", "ts_us", "event_seq"], kind="mergesort").reset_index(drop=True)
        kept = getattr(self, "_kept_mask", None)
        if kept is not None and len(kept) == len(frame):  # same order as the flat history arrays
            frame = frame[kept].reset_index(drop=True)
        return frame

    @property
    def cache_dir(self) -> Path:
        path = self.root / "cache" / self.split_hash
        path.mkdir(parents=True, exist_ok=True)
        return path

    def warm_users(self) -> np.ndarray:
        return np.flatnonzero(self.user_lengths > 0).astype(np.int64)

    def warm_item_mask(self) -> np.ndarray:
        """True for items with at least one pre-test event. Index 0 (padding) is False."""
        mask = self.item_pop > 0
        mask[0] = False
        return mask

    # ----- time-aware views (tunable settings `train_window_days` and `decay_half_life_days`) -----
    def event_age_days(self) -> np.ndarray:
        """Age of every pre-test event at the cutoff, in days (same order as the flat history arrays)."""
        return (int(self.meta["test_start_us"]) - np.asarray(self._ts, dtype=np.int64)) / DAY_US

    def event_weights(self, half_life_days: float | None = None) -> np.ndarray:
        """Per-event weight 2^(-age / half_life): 1 at the cutoff, 0.5 one half-life earlier. All 1 without decay."""
        if not half_life_days:
            return np.ones(len(self._items), dtype=np.float32)
        return np.power(2.0, -self.event_age_days() / float(half_life_days)).astype(np.float32)

    def weighted_matrix(self, half_life_days: float | None = None, binary: bool = True) -> sp.csr_matrix:
        """User x item matrix of time-decayed weights, shape [n_users + 1, n_items + 1].

        binary=True: each (user, item) cell holds the weight of the user's MOST RECENT event with that item
        (so it equals `seen` without decay). binary=False: the sum of the event weights (decayed counts).
        """
        if not half_life_days:
            return self.seen if binary else self.interaction_counts
        lengths = self.user_lengths
        rows = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
        cols = np.asarray(self._items, dtype=np.int64)
        weights = self.event_weights(half_life_days)
        shape = (self.n_users + 1, self.n_items + 1)
        if not binary:
            matrix = sp.csr_matrix((weights, (rows, cols)), shape=shape)
            matrix.sum_duplicates()
            return matrix
        order = np.lexsort((weights, cols, rows))  # per (row, col), the largest weight comes last
        rows, cols, weights = rows[order], cols[order], weights[order]
        last = np.ones(len(rows), dtype=bool)
        last[:-1] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
        return sp.csr_matrix((weights[last], (rows[last], cols[last])), shape=shape)

    def restrict(self, window_days: float | None, keep_last: int = 10) -> "TrainView":
        """A view of the last `window_days` days before the cutoff, plus each user's `keep_last` most recent events.

        Used for fitting only (the evaluator keeps the full history for masking seen items). Keeping the last
        few events of every user means nobody loses their whole profile when the window is short.
        """
        if not window_days:
            return self
        offsets = np.asarray(self._offsets, dtype=np.int64)
        lengths = np.diff(offsets)
        owner = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
        from_end = offsets[owner + 1] - 1 - np.arange(len(owner), dtype=np.int64)  # 0 = the user's latest event
        keep = (self.event_age_days() <= float(window_days)) | (from_end < int(keep_last))
        view = copy.copy(self)
        for cached in ("user_lengths", "interaction_counts", "seen"):
            view.__dict__.pop(cached, None)
        view._items = np.asarray(self._items)[keep]
        view._ts = np.asarray(self._ts)[keep]
        new_offsets = np.zeros_like(offsets)
        new_offsets[1:] = np.cumsum(np.bincount(owner[keep], minlength=len(lengths)))
        view._offsets = new_offsets
        recent = view.event_age_days() <= RECENT_DAYS
        view.item_pop = np.bincount(view._items, minlength=self.n_items + 1).astype(np.int64)
        view.item_recent_pop = np.bincount(view._items[recent], minlength=self.n_items + 1).astype(np.int64)
        view.meta = {**self.meta, "train_window_days": float(window_days), "train_window_keep_last": int(keep_last)}
        view._kept_mask = keep
        return view
