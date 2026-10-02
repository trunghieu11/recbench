"""Shared training loop for the in-repo model bodies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

from recbench.protocol import Explanation, Recommender, Unsupported


@dataclass
class TrainConfig:
    max_steps: int = 200
    dim: int = 16
    layers: int = 1
    seq_len: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    device: str = "cpu"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrainConfig":
        device = raw.get("device")
        if device is None:
            if raw.get("model_config") == "gpu_full" and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        known = {field: raw[field] for field in cls.__dataclass_fields__ if field in raw and field != "device"}
        return cls(device=device, **known)


def nhead(dim: int) -> int:
    for heads in (4, 2, 1):
        if dim % heads == 0:
            return heads
    return 1


class TorchMethod(Recommender):
    loss = "bpr"

    def build_model(self, n_users: int, n_items: int, cfg: TrainConfig) -> nn.Module:
        raise NotImplementedError

    def loss_batch(self, model: nn.Module, batch: dict[str, torch.Tensor], cfg: TrainConfig) -> torch.Tensor:
        raise NotImplementedError

    def score_batch(
        self,
        model: nn.Module,
        users: torch.Tensor,
        items: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def fit(self, store: Any, config: dict[str, Any]) -> None:
        if self.spec.requires_images and not store.images_available():
            raise Unsupported(f"{self.spec.name} needs images")
        cfg = TrainConfig.from_dict(config)
        torch.manual_seed(42)
        np.random.seed(42)
        n_users = int(store.meta["n_users"])
        n_items = int(store.meta["n_items"])
        model = self.build_model(n_users, n_items, cfg).to(cfg.device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        steps = 0
        model.train()
        while steps < cfg.max_steps:
            progressed = False
            for batch in iter_batches(self.loss, store, cfg):
                progressed = True
                batch = {key: value.to(cfg.device) for key, value in batch.items()}
                loss = self.loss_batch(model, batch, cfg)
                opt.zero_grad()
                loss.backward()
                opt.step()
                steps += 1
                if steps >= cfg.max_steps:
                    break
            if not progressed:
                break
        self.model = model.cpu()
        self.cfg = cfg
        self.n_users = n_users
        self.n_items = n_items
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
        if not self.spec.can_score_candidates or self.spec.text_only:
            raise Unsupported(f"{self.spec.name} does not score the candidate file")
        cfg = self.cfg
        hist = store.history_matrix(cfg.seq_len)
        frame = candidates
        users = frame["user_idx"].to_numpy()
        items = frame["item_idx"].to_numpy()
        scores = np.zeros(len(frame), dtype=np.float32)
        self.model.eval()
        device = cfg.device
        self.model.to(device)
        with torch.no_grad():
            for start in range(0, len(frame), 4096):
                stop = start + 4096
                user_t = torch.tensor(users[start:stop], dtype=torch.long, device=device)
                item_t = torch.tensor(items[start:stop], dtype=torch.long, device=device)
                hist_t = torch.tensor(np.asarray(hist[users[start:stop]]), dtype=torch.long, device=device)
                scores[start:stop] = self.score_batch(self.model, user_t, item_t, hist_t).detach().cpu().numpy()
        self.model.cpu()
        out = frame.copy()
        out["score"] = scores
        return out

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state": self.model.state_dict(),
                "cfg": self.cfg.__dict__,
                "n_users": self.n_users,
                "n_items": self.n_items,
                "loss": self.loss,
            },
            path,
        )

    def load(self, path: str) -> None:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.cfg = TrainConfig(**blob["cfg"])
        self.n_users = blob["n_users"]
        self.n_items = blob["n_items"]
        self.model = self.build_model(self.n_users, self.n_items, self.cfg)
        self.model.load_state_dict(blob["state"])
        self.model.eval()

    def recommend(self, user_id: str, k: int = 10) -> list[dict[str, Any]]:
        uid = self.user_to.get(str(user_id))
        hist = self.store.history_matrix(self.cfg.seq_len)
        n_items = self.n_items
        item_ids = np.arange(1, n_items + 1)
        if uid is None:
            counts = _item_counts(self.store)
            order = sorted(item_ids, key=lambda i: counts.get(int(i), 0), reverse=True)[:k]
            return [
                {
                    "item_id": self.idx_item.get(int(i), str(i)),
                    "score": float(counts.get(int(i), 0)),
                    "explanation": self._explain(user_id, self.idx_item.get(int(i), str(i)), cold=True),
                }
                for i in order
            ]
        users = np.full(len(item_ids), uid)
        frame_users = torch.tensor(users, dtype=torch.long)
        frame_items = torch.tensor(item_ids, dtype=torch.long)
        history = torch.tensor(np.asarray(hist[users]), dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            scores = self.score_batch(self.model, frame_users, frame_items, history).numpy()
        seen = set(int(x) for x in hist[uid] if int(x) != 0)
        order = [i for i in np.argsort(-scores) if int(item_ids[i]) not in seen][:k]
        return [
            {
                "item_id": self.idx_item.get(int(item_ids[i]), str(item_ids[i])),
                "score": float(scores[i]),
                "explanation": self._explain(user_id, self.idx_item.get(int(item_ids[i]), str(item_ids[i]))),
            }
            for i in order
        ]

    def similar_items(self, item_id: str, k: int = 10) -> list[dict[str, Any]]:
        if not hasattr(self.model, "item"):
            raise Unsupported(f"{self.spec.name} has no item embedding")
        idx = self.item_to.get(str(item_id))
        if idx is None:
            return []
        weight = self.model.item.weight.detach()
        query = weight[idx]
        scores = torch.mv(weight, query)
        scores[0] = -1e9
        scores[idx] = -1e9
        top = torch.topk(scores, k=min(k, scores.numel() - 2)).indices.tolist()
        return [{"item_id": self.idx_item.get(int(i), str(i)), "score": float(scores[i])} for i in top]

    def explain_global(self) -> Explanation | None:
        return Explanation(kind="global", text=self.spec.upstream or self.spec.name, evidence=[])

    def explain_local(self, user_id: str, item_ids: list[str]) -> list[Explanation]:
        return [self._explain(user_id, item_id) for item_id in item_ids]

    def _explain(self, user_id: str, item_id: str, cold: bool = False) -> Explanation:
        if cold:
            text = f"{item_id} is a frequent training item. {user_id} was not in the training split."
        else:
            text = f"{self.spec.name} scores {item_id} from the cutoff-truncated history of {user_id}."
        return Explanation(kind="local", text=text, evidence=[{"item_id": item_id}])

    def predict_rating(self, frame: Any):
        if "rating" not in {task.value for task in self.spec.tasks}:
            raise Unsupported(f"{self.spec.name} has no rating head")
        scored = self.score_candidates(self.store, frame)
        return 1 + 4 * torch.sigmoid(torch.tensor(scored["score"].to_numpy())).numpy()

    def predict_ctr(self, frame: Any):
        scored = self.score_candidates(self.store, frame)
        return torch.sigmoid(torch.tensor(scored["score"].to_numpy())).numpy()


def iter_batches(loss: str, store: Any, cfg: TrainConfig) -> Iterator[dict[str, torch.Tensor]]:
    if loss == "bpr":
        yield from _edge_epoch(store, cfg)
    elif loss == "bce":
        yield from _candidate_epoch(store, cfg)
    else:
        yield from _sequence_epoch(store, cfg)


def _edge_epoch(store: Any, cfg: TrainConfig) -> Iterator[dict[str, torch.Tensor]]:
    table = pq.read_table(store.train_path, columns=["user_idx", "item_idx"]).to_pandas()
    if table.empty:
        return
    users = table["user_idx"].to_numpy()
    items = table["item_idx"].to_numpy()
    rng = np.random.default_rng(42)
    n_items = int(store.meta["n_items"])
    idx = rng.permutation(len(users))
    for start in range(0, len(idx), cfg.batch_size):
        chosen = idx[start : start + cfg.batch_size]
        neg = rng.integers(1, n_items + 1, size=len(chosen))
        yield {
            "users": torch.tensor(users[chosen], dtype=torch.long),
            "pos": torch.tensor(items[chosen], dtype=torch.long),
            "neg": torch.tensor(neg, dtype=torch.long),
        }


def _candidate_epoch(store: Any, cfg: TrainConfig) -> Iterator[dict[str, torch.Tensor]]:
    hist = store.history_matrix(cfg.seq_len)
    parquet = pq.ParquetFile(store.candidates_path)
    for batch in parquet.iter_batches(batch_size=cfg.batch_size):
        frame = batch.to_pandas()
        users = frame["user_idx"].to_numpy()
        yield {
            "users": torch.tensor(users, dtype=torch.long),
            "items": torch.tensor(frame["item_idx"].to_numpy(), dtype=torch.long),
            "labels": torch.tensor(frame["label"].to_numpy(), dtype=torch.float32),
            "history": torch.tensor(np.asarray(hist[users]), dtype=torch.long),
        }


def _sequence_epoch(store: Any, cfg: TrainConfig) -> Iterator[dict[str, torch.Tensor]]:
    hist = store.history_matrix(cfg.seq_len)
    table = pq.read_table(store.sequences_path, columns=["user_idx", "target_item"]).to_pandas()
    if table.empty:
        return
    users = table["user_idx"].to_numpy()
    targets = table["target_item"].to_numpy()
    for start in range(0, len(users), cfg.batch_size):
        sl = slice(start, start + cfg.batch_size)
        yield {
            "users": torch.tensor(users[sl], dtype=torch.long),
            "items": torch.tensor(targets[sl], dtype=torch.long),
            "history": torch.tensor(np.asarray(hist[users[sl]]), dtype=torch.long),
        }


def _item_counts(store: Any) -> dict[int, int]:
    frame = pq.read_table(store.train_path, columns=["item_idx"]).to_pandas()
    return frame["item_idx"].value_counts().astype(int).to_dict()


def sampled_softmax(hidden: torch.Tensor, weight: torch.Tensor, target: torch.Tensor, n_items: int) -> torch.Tensor:
    neg = torch.randint(1, n_items + 1, (target.shape[0], 20), device=target.device)
    cands = torch.cat([target.unsqueeze(1), neg], dim=1)
    emb = weight[cands]
    logits = (emb * hidden.unsqueeze(1)).sum(-1)
    labels = torch.zeros(target.shape[0], dtype=torch.long, device=target.device)
    return torch.nn.functional.cross_entropy(logits, labels)
