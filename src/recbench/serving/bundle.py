"""Serving bundles: precomputed top-K lists, so the online API needs no model code (and no torch).

A bundle for one (dataset, tier, method) is a folder with:
    manifest.json      what it is, when and how it was built
    users.parquet      user_id -> row in topk.npy (recently active users only)
    topk.npy           int32 [n_users, k] item indices, best first (seen items removed)
    scores.npy         float32 [n_users, k] model scores for those items
    items.parquet      item_idx -> item_id, text (for display)
    popular.npy        int32 [k] fallback list for unknown users (recent popularity)

Trade-off: lookups are fast and cheap (a numpy row read), but lists are as fresh
as the last export. Real-time models would score online instead.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recbench.protocol import NEG_INF, PROTOCOL_VERSION, top_k


def export_bundle(method: Any, data: Any, out: Path, k: int = 100, max_users: int = 20_000, batch: int = 512) -> Path:
    """Score the most recently active warm users and write a bundle folder (replacing any old one)."""
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    offsets = np.asarray(data._offsets)
    warm = data.warm_users()
    last_seen = np.asarray(data._ts)[offsets[warm + 1] - 1] if len(warm) else np.zeros(0)
    users = np.sort(warm[np.argsort(-last_seen, kind="stable")[:max_users]])
    seq_len = int(getattr(method, "seq_len", 50))
    cold_items = ~data.warm_item_mask()
    cold_items[0] = False
    lists = np.zeros((len(users), k), dtype=np.int32)
    values = np.zeros((len(users), k), dtype=np.float32)
    began = time.perf_counter()
    for start in range(0, len(users), batch):
        chunk = users[start : start + batch]
        scores = method.full_scores(chunk, data.history_batch(chunk, seq_len))
        rows, cols = data.seen[chunk].nonzero()
        scores[rows, cols] = NEG_INF
        scores[:, 0] = NEG_INF
        if not method.spec.scores_cold_items:
            scores[:, cold_items] = NEG_INF
        items, top_scores = top_k(scores, k)
        lists[start : start + len(chunk), : items.shape[1]] = items
        values[start : start + len(chunk), : items.shape[1]] = top_scores
    popular = np.argsort(-(data.item_recent_pop + data.item_pop / (data.item_pop.max() + 1.0)), kind="stable")
    popular = popular[popular > 0][:k].astype(np.int32)
    np.save(tmp / "topk.npy", lists)
    np.save(tmp / "scores.npy", values)
    np.save(tmp / "popular.npy", popular)
    pd.DataFrame({"user_id": data.user_ids[users].astype(str), "row": np.arange(len(users), dtype=np.int32)}).to_parquet(tmp / "users.parquet", index=False)
    pd.DataFrame({"item_idx": np.arange(data.n_items + 1), "item_id": data.item_ids.astype(str), "text": data.item_text.astype(str)}).to_parquet(
        tmp / "items.parquet", index=False
    )
    manifest = {
        "dataset": data.dataset,
        "tier": data.tier,
        "method": method.spec.name,
        "protocol_version": PROTOCOL_VERSION,
        "split_hash": data.split_hash,
        "k": k,
        "n_users": int(len(users)),
        "n_items": int(data.n_items),
        "export_seconds": round(time.perf_counter() - began, 3),
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "freshness": f"lists reflect events before {data.meta.get('test_start', '?')} UTC",
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return out


class Bundle:
    """Read side, used by the API. Loads only numpy and parquet files."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / "manifest.json").read_text())
        self.topk = np.load(self.folder / "topk.npy", mmap_mode="r")
        self.scores = np.load(self.folder / "scores.npy", mmap_mode="r")
        self.popular = np.load(self.folder / "popular.npy")
        users = pd.read_parquet(self.folder / "users.parquet")
        self.row_of = dict(zip(users["user_id"].astype(str), users["row"].astype(int)))
        items = pd.read_parquet(self.folder / "items.parquet")
        self.item_id = items["item_id"].astype(str).to_numpy()
        self.item_text = items["text"].astype(str).to_numpy()

    def recommend(self, user_id: str, k: int) -> dict[str, Any]:
        k = max(1, min(int(k), int(self.manifest["k"])))
        row = self.row_of.get(str(user_id))
        if row is None:
            items, scores, fallback = self.popular[:k], [None] * k, True
        else:
            items = np.asarray(self.topk[row, :k])
            scores = [float(s) for s in self.scores[row, :k]]
            fallback = False
        recs = [
            {"item_id": self.item_id[i], "title": self.item_text[i], "score": s}
            for i, s in zip(items, scores)
            if i > 0
        ]
        return {"user_id": str(user_id), "fallback": fallback, "recommendations": recs}
