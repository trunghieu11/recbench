"""XSimGCL using SELFRec's XSimGCL_Encoder and official BPR / InfoNCE losses.

Commit: third_party/SELFRec 5b0229423cb1c727e85a704d63e460368c8b9dde
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp
import torch

from recbench.methods._common import TrainConfig
from recbench.protocol import Explanation, MethodSpec, Recommender, Task
from recbench.registry import register_method

_SELFREC = Path(__file__).resolve().parents[3] / "third_party" / "SELFRec"


@contextmanager
def _cpu_cuda():
    original = torch.Tensor.cuda

    def _stay(self, *args, **kwargs):
        return self

    torch.Tensor.cuda = _stay
    try:
        yield
    finally:
        torch.Tensor.cuda = original


def _encoder_class():
    if str(_SELFREC) not in sys.path:
        sys.path.insert(0, str(_SELFREC))
    from model.graph.XSimGCL import XSimGCL_Encoder
    from util.loss_torch import InfoNCE, bpr_loss

    return XSimGCL_Encoder, bpr_loss, InfoNCE


def _normalized_adjacency(store) -> tuple[sp.csr_matrix, int, int]:
    train = pq.read_table(store.train_path, columns=["user_idx", "item_idx"]).to_pandas()
    n_users = int(store.meta["n_users"]) + 1
    n_items = int(store.meta["n_items"]) + 1
    rows = np.concatenate([train["user_idx"].to_numpy(), train["item_idx"].to_numpy() + n_users])
    cols = np.concatenate([train["item_idx"].to_numpy() + n_users, train["user_idx"].to_numpy()])
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n_users + n_items, n_users + n_items)).tocsr()
    degree = np.array(adj.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    scale = sp.diags(np.power(degree, -0.5))
    return (scale @ adj @ scale).tocsr(), n_users, n_items


class _Data:
    def __init__(self, n_users: int, n_items: int, adj: sp.csr_matrix):
        self.user_num = n_users
        self.item_num = n_items
        self.norm_adj = adj


@register_method
class XSimGCL(Recommender):
    spec = MethodSpec(
        name="xsimgcl",
        tasks={Task.topn, Task.similar_items},
        feedback={"implicit", "explicit"},
        cost_band="medium",
        upstream="Coder-Yu/SELFRec XSimGCL_Encoder @ 5b022942",
    )

    def fit(self, store: Any, config: dict[str, Any]) -> None:
        cfg = TrainConfig.from_dict(config)
        encoder_cls, bpr_loss, info_nce = _encoder_class()
        adj, n_users, n_items = _normalized_adjacency(store)
        data = _Data(n_users, n_items, adj)
        with _cpu_cuda():
            encoder = encoder_cls(data, cfg.dim, 0.1, max(cfg.layers, 1), 1)
        encoder.sparse_norm_adj = encoder.sparse_norm_adj.cpu()
        optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.lr)
        train = pq.read_table(store.train_path, columns=["user_idx", "item_idx"]).to_pandas()
        users = train["user_idx"].to_numpy()
        items = train["item_idx"].to_numpy()
        rng = np.random.default_rng(42)
        encoder.train()
        steps = 0
        while steps < cfg.max_steps and len(users):
            choice = rng.integers(0, len(users), size=min(cfg.batch_size, len(users)))
            user_idx = torch.tensor(users[choice], dtype=torch.long)
            pos_idx = torch.tensor(items[choice], dtype=torch.long)
            neg_idx = torch.tensor(rng.integers(1, n_items, size=len(choice)), dtype=torch.long)
            with _cpu_cuda():
                rec_user, rec_item, cl_user, cl_item = encoder(True)
            loss = bpr_loss(rec_user[user_idx], rec_item[pos_idx], rec_item[neg_idx])
            loss = loss + 0.1 * (
                info_nce(rec_user[user_idx], cl_user[user_idx], 0.2)
                + info_nce(rec_item[pos_idx], cl_item[pos_idx], 0.2)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1
        encoder.eval()
        with torch.no_grad(), _cpu_cuda():
            self.user_emb, self.item_emb = encoder(False)
        self.encoder = encoder
        self.cfg = cfg
        self._bind(store)

    def _bind(self, store: Any) -> None:
        user_to, idx_user, item_to, idx_item = store.id_maps()
        self.user_to = user_to
        self.idx_user = idx_user
        self.item_to = item_to
        self.idx_item = idx_item
        self.store = store

    def score_candidates(self, store: Any, candidates: Any = None):
        if candidates is None:
            candidates = pq.read_table(store.candidates_path).to_pandas()
        frame = candidates.copy()
        users = torch.tensor(frame["user_idx"].to_numpy().astype("int64"), dtype=torch.long)
        items = torch.tensor(frame["item_idx"].to_numpy().astype("int64"), dtype=torch.long)
        with torch.no_grad():
            scores = (self.user_emb[users] * self.item_emb[items]).sum(-1)
        frame["score"] = scores.cpu().numpy()
        return frame

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.encoder.state_dict()}, path)

    def load(self, path: str) -> None:
        raise RuntimeError("XSimGCL reload requires the SELFRec graph built at fit time.")

    def explain_global(self) -> Explanation | None:
        return Explanation("global", self.spec.upstream, [])

    def explain_local(self, user_id: str, item_ids: list[str]) -> list[Explanation]:
        return [
            Explanation("local", f"XSimGCL embedding of {user_id} is close to {item_id}.", [{"item_id": item_id}])
            for item_id in item_ids
        ]

    def recommend(self, user_id: str, k: int = 10) -> list[dict[str, Any]]:
        uid = self.user_to.get(str(user_id))
        if uid is None:
            return []
        scores = torch.mv(self.item_emb, self.user_emb[uid])
        top = torch.topk(scores, k=min(k, scores.numel() - 1)).indices.tolist()
        return [{"item_id": self.idx_item.get(int(i), str(i)), "score": float(scores[i])} for i in top if int(i) != 0]
