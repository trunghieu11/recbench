"""On-disk split produced by the warehouse pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow.parquet as pq


@dataclass
class SplitStore:
    root: Path
    dataset: str
    tier: str

    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    @property
    def meta(self) -> dict[str, Any]:
        return json.loads(self.meta_path.read_text())

    @property
    def train_path(self) -> Path:
        return self.root / "train.parquet"

    @property
    def valid_path(self) -> Path:
        return self.root / "valid.parquet"

    @property
    def test_path(self) -> Path:
        return self.root / "test.parquet"

    @property
    def items_path(self) -> Path:
        return self.root / "items.parquet"

    @property
    def users_path(self) -> Path:
        return self.root / "users.parquet"

    @property
    def sequences_path(self) -> Path:
        return self.root / "sequences.parquet"

    @property
    def candidates_path(self) -> Path:
        return self.root / "candidates.parquet"

    def connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect()

    def images_available(self) -> bool:
        if not self.items_path.exists():
            return False
        frame = pq.read_table(self.items_path, columns=["image_path"]).to_pandas()
        paths = [p for p in frame["image_path"].dropna().astype(str) if p and p != "None"]
        if not paths:
            return False
        for path in paths[:30]:
            if Path(path).is_file():
                return True
        return False

    def history_matrix(self, seq_len: int) -> np.ndarray:
        """Right-aligned histories. Index 0 is padding. Built by streaming the sequence file."""
        meta = self.meta
        n_users = int(meta["n_users"])
        cache = self.root / f"history_{seq_len}.npy"
        if cache.exists():
            return np.load(cache, mmap_mode="r")
        matrix = np.zeros((n_users + 1, seq_len), dtype=np.int32)
        if not self.sequences_path.exists():
            np.save(cache, matrix)
            return np.load(cache, mmap_mode="r")
        parquet = pq.ParquetFile(self.sequences_path)
        for batch in parquet.iter_batches(batch_size=4096, columns=["user_idx", "history"]):
            users = batch.column("user_idx").to_pylist()
            histories = batch.column("history").to_pylist()
            for user, hist in zip(users, histories):
                if not hist:
                    continue
                tail = [int(x) for x in hist[-seq_len:]]
                matrix[int(user), seq_len - len(tail) :] = tail
        np.save(cache, matrix)
        return np.load(cache, mmap_mode="r")

    def item_categories(self) -> dict[int, str]:
        frame = pq.read_table(self.items_path, columns=["item_idx", "category"]).to_pandas()
        return {int(i): str(c or "") for i, c in zip(frame["item_idx"], frame["category"])}

    def item_text(self) -> dict[int, str]:
        frame = pq.read_table(self.items_path, columns=["item_idx", "text"]).to_pandas()
        return {int(i): str(t or "") for i, t in zip(frame["item_idx"], frame["text"])}

    def id_maps(self) -> tuple[dict[str, int], dict[int, str], dict[str, int], dict[int, str]]:
        items = pq.read_table(self.items_path, columns=["item_id", "item_idx"]).to_pandas()
        users = pq.read_table(self.users_path, columns=["user_id", "user_idx"]).to_pandas()
        item_to = {str(i): int(x) for i, x in zip(items["item_id"], items["item_idx"])}
        idx_item = {int(x): str(i) for i, x in zip(items["item_id"], items["item_idx"])}
        user_to = {str(i): int(x) for i, x in zip(users["user_id"], users["user_idx"])}
        idx_user = {int(x): str(i) for i, x in zip(users["user_id"], users["user_idx"])}
        return user_to, idx_user, item_to, idx_item
