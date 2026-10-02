"""Graph collaborative filtering through SELFRec's encoders: LightGCN and XSimGCL.

Both treat users and items as nodes of one bipartite graph and smooth their
embeddings over the normalised adjacency matrix. The training loops mirror
SELFRec's (model/graph/LightGCN.py and XSimGCL.py at commit 5b022942), with a
step budget instead of epochs and device-aware tensors instead of .cuda().
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch

from recbench.data import HistoryBatch, TrainView
from recbench.methods._torch import EmbeddingRecommender, resolve_device, sample_negatives, warm_items
from recbench.protocol import MethodSpec, Task, Unsupported
from recbench.registry import register_method

SELFREC = Path(__file__).resolve().parents[3] / "third_party" / "SELFRec"
GRAPH_TASKS = {Task.topn, Task.sequential, Task.similar_items}


def _selfrec():
    if not (SELFREC / "model").is_dir():
        raise Unsupported("third_party/SELFRec is missing; run scripts/fetch_third_party.sh")
    if str(SELFREC) not in sys.path:
        sys.path.insert(0, str(SELFREC))
    from model.graph.LightGCN import LGCN_Encoder
    from model.graph.XSimGCL import XSimGCL_Encoder
    from util.loss_torch import InfoNCE, bpr_loss, l2_reg_loss

    return LGCN_Encoder, XSimGCL_Encoder, bpr_loss, l2_reg_loss, InfoNCE


@contextmanager
def on_device(device: torch.device):
    """SELFRec hard-codes tensor.cuda(); redirect it to the chosen device (CPU on a laptop)."""
    original = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *args, **kwargs: self.to(device)
    try:
        yield
    finally:
        torch.Tensor.cuda = original


class _GraphData:
    """The attributes SELFRec encoders read: user_num, item_num, norm_adj."""

    def __init__(self, seen: sp.csr_matrix):
        n_users, n_items = seen.shape
        self.user_num, self.item_num = n_users, n_items
        coo = seen.tocoo()
        rows = np.concatenate([coo.row, coo.col + n_users])
        cols = np.concatenate([coo.col + n_users, coo.row])
        adj = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_users + n_items,) * 2).tocsr()
        degree = np.asarray(adj.sum(axis=1)).ravel()
        scale = sp.diags(np.power(np.where(degree > 0, degree, 1.0), -0.5))
        self.norm_adj = (scale @ adj @ scale).tocsr().astype(np.float32)


class _GraphMethod(EmbeddingRecommender):
    def _prepare(self, data: TrainView, cfg: dict[str, Any]):
        self.bind(data)
        self.device = resolve_device(cfg)
        self.graph = _GraphData(data.seen)
        self.seen = data.seen
        self.events = data.events()
        self.pool = warm_items(data)
        self.rng = np.random.default_rng(int(cfg.get("seed", 42)))
        self.batch_size = int(cfg.get("graph_batch_size", 2048))

    def _batch(self):
        pick = self.rng.integers(0, len(self.events), size=self.batch_size)
        users = self.events["user_idx"].to_numpy()[pick]
        pos = self.events["item_idx"].to_numpy()[pick]
        neg = sample_negatives(self.seen, users, 1, self.pool, self.rng)[:, 0]
        as_t = lambda a: torch.as_tensor(a, dtype=torch.long, device=self.device)  # noqa: E731
        return as_t(users), as_t(pos), as_t(neg)

    def user_vectors(self, users: np.ndarray, hist: HistoryBatch) -> torch.Tensor:
        return self.user_emb[torch.as_tensor(users, dtype=torch.long)]

    def item_matrix(self) -> torch.Tensor:
        return self.item_emb


@register_method
class LightGCN(_GraphMethod):
    spec = MethodSpec(
        name="lightgcn",
        tasks=GRAPH_TASKS,
        needs_torch=True,
        upstream="SELFRec LGCN_Encoder @ 5b022942",
        cost_band="medium",
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self._prepare(data, cfg)
        encoder_cls, _, bpr_loss, l2_reg_loss, _ = _selfrec()
        dim, layers, reg = int(cfg.get("dim", 64)), int(cfg.get("graph_layers", 2)), float(cfg.get("graph_reg", 1e-4))
        with on_device(self.device):
            model = encoder_cls(self.graph, dim, layers).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 1e-3)))
            for _ in range(int(cfg.get("max_steps", 400))):
                users, pos, neg = self._batch()
                user_all, item_all = model()
                loss = bpr_loss(user_all[users], item_all[pos], item_all[neg]) + l2_reg_loss(
                    reg, model.embedding_dict["user_emb"][users], model.embedding_dict["item_emb"][pos], model.embedding_dict["item_emb"][neg]
                ) / self.batch_size
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                user_all, item_all = model()
        self.user_emb, self.item_emb = user_all.detach().cpu(), item_all.detach().cpu()
        self.fit_info = {"final_loss": float(loss.detach())}


@register_method
class XSimGCL(_GraphMethod):
    spec = MethodSpec(
        name="xsimgcl",
        tasks=GRAPH_TASKS,
        needs_torch=True,
        upstream="SELFRec XSimGCL_Encoder @ 5b022942",
        cost_band="medium",
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self._prepare(data, cfg)
        _, encoder_cls, bpr_loss, l2_reg_loss, info_nce = _selfrec()
        dim = int(cfg.get("dim", 64))
        layers = int(cfg.get("graph_layers", 2))
        eps, cl_rate, temp = float(cfg.get("xsim_eps", 0.2)), float(cfg.get("xsim_lambda", 0.2)), float(cfg.get("xsim_tau", 0.15))
        layer_cl, reg = int(cfg.get("xsim_layer_cl", 1)), float(cfg.get("graph_reg", 1e-4))
        with on_device(self.device):
            model = encoder_cls(self.graph, dim, eps, layers, layer_cl).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 1e-3)))
            for _ in range(int(cfg.get("max_steps", 400))):
                users, pos, neg = self._batch()
                rec_u, rec_i, cl_u, cl_i = model(True)
                u_emb, p_emb, n_emb = rec_u[users], rec_i[pos], rec_i[neg]
                cl = cl_rate * (
                    info_nce(rec_u[users.unique()], cl_u[users.unique()], temp) + info_nce(rec_i[pos.unique()], cl_i[pos.unique()], temp)
                )
                loss = bpr_loss(u_emb, p_emb, n_emb) + l2_reg_loss(reg, u_emb, p_emb) + cl
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                user_all, item_all = model()
        self.user_emb, self.item_emb = user_all.detach().cpu(), item_all.detach().cpu()
        self.fit_info = {"final_loss": float(loss.detach())}
