"""Generative recommender that scores candidate ids from item text.

Smoke uses a hashed text tower (P5-style scoring without a download). The GPU
config records a 7B/8B QLoRA model name; that download is not required to rank.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F

from recbench.methods._common import TorchMethod
from recbench.protocol import Explanation, MethodSpec, Task
from recbench.registry import register_method

GPU_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _hash_vec(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = str(text).lower().split() or ["empty"]
    for token in tokens:
        digest = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[digest % dim] += 1.0
    norm = np.linalg.norm(vec) or 1.0
    return vec / norm


class _Net(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, content: torch.Tensor):
        super().__init__()
        self.user = nn.Embedding(n_users + 1, dim, padding_idx=0)
        self.register_buffer("content", content)
        self.prompt = "User history: {history}\nCandidate: {item}\nRecommend it?"

    def bpr(self, users: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        user = self.user(users)
        pos_s = (user * self.content[pos]).sum(-1)
        neg_s = (user * self.content[neg]).sum(-1)
        return -F.logsigmoid(pos_s - neg_s).mean()

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user(users) * self.content[items]).sum(-1)


@register_method
class GenerativeLLM(TorchMethod):
    spec = MethodSpec(
        name="generative_llm",
        tasks={Task.topn, Task.sequential, Task.rating},
        feedback={"implicit", "explicit"},
        requires_side_features=True,
        can_score_candidates=True,
        text_only=False,
        cost_band="medium",
        upstream=f"P5/TALLRec-style scoring. GPU preset names {GPU_MODEL} with 4-bit QLoRA; smoke uses a text hash tower that scores candidates.",
    )
    loss = "bpr"

    def build_model(self, n_users: int, n_items: int, cfg):
        content = torch.zeros(n_items + 1, cfg.dim)
        if getattr(self, "store", None) is not None:
            items = pq.read_table(self.store.items_path, columns=["item_idx", "text", "category"]).to_pandas()
            for idx, text, category in zip(items["item_idx"], items["text"], items["category"]):
                content[int(idx)] = torch.tensor(_hash_vec(f"{text} {category}", cfg.dim))
        return _Net(n_users, n_items, cfg.dim, content)

    def fit(self, store, config):
        self.store = store
        self.hf_model = GPU_MODEL if config.get("model_config") == "gpu_full" else "hash-tower"
        super().fit(store, config)

    def loss_batch(self, model, batch, cfg) -> torch.Tensor:
        return model.bpr(batch["users"], batch["pos"], batch["neg"])

    def score_batch(self, model, users, items, history) -> torch.Tensor:
        return model.score(users, items)

    def explain_global(self) -> Explanation:
        model_name = getattr(self, "hf_model", "hash-tower")
        return Explanation("global", f"Prompt template ranks candidates. Backbone: {model_name}.", [{"backbone": model_name}])

    def _explain(self, user_id: str, item_id: str, cold: bool = False) -> Explanation:
        return Explanation(
            "local",
            f"Because the history of {user_id} shares text with {item_id}, the generative scorer ranks it.",
            [{"item_id": item_id}],
        )
