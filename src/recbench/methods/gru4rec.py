"""GRU4Rec (Hidasi et al. 2016, 2018) through the authors' official PyTorch implementation.

The official code is fetched (never vendored) into third_party/GRU4Rec_PyTorch_Official by
scripts/fetch_third_party.sh. Its licence allows research and education; commercial use needs the author's
permission. Third-party re-implementations of GRU4Rec are known to underperform (Hidasi & Czapp 2023),
which is why recbench uses the original.

Adaptation to recbench's task: each user's pre-test history is one long sequence (the "session"), trained
with GRU4Rec's session-parallel mini-batches. To score a user, the recurrent cells read the user's latest
`seq_len` items in order, and the final hidden state is scored against every item.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recbench.data import HistoryBatch, TrainView
from recbench.protocol import NEG_INF, MethodSpec, Recommender, Task, Unsupported
from recbench.registry import register_method

OFFICIAL = Path(__file__).resolve().parents[3] / "third_party" / "GRU4Rec_PyTorch_Official"


def _official():
    if not (OFFICIAL / "gru4rec_pytorch.py").is_file():
        raise Unsupported("third_party/GRU4Rec_PyTorch_Official is missing; run scripts/fetch_third_party.sh")
    if str(OFFICIAL) not in sys.path:
        sys.path.insert(0, str(OFFICIAL))
    import gru4rec_pytorch

    return gru4rec_pytorch


@register_method
class GRU4Rec(Recommender):
    spec = MethodSpec(
        name="gru4rec",
        tasks={Task.topn, Task.sequential, Task.session},
        uses_history=True,
        needs_torch=True,
        upstream="official GRU4Rec_PyTorch_Official (pinned commit; research/education licence)",
        cost_band="low",
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        import torch

        official = _official()
        self.bind(data)
        # The official code uses sparse embeddings and its own optimiser: CUDA or CPU (not Apple MPS).
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
        self.seq_len = int(cfg.get("seq_len", 50))
        events = data.events()
        frame = pd.DataFrame({"SessionId": events["user_idx"].to_numpy(), "ItemId": events["item_idx"].to_numpy(),
                              "Time": events["ts_us"].to_numpy()})
        hidden = int(cfg.get("gru4rec_hidden", 224))
        loss = str(cfg.get("gru4rec_loss", "bpr-max"))
        model = official.GRU4Rec(
            layers=[hidden], loss=loss, batch_size=int(cfg.get("gru4rec_batch_size", 80)),
            dropout_p_embed=float(cfg.get("gru4rec_dropout_embed", 0.5)), dropout_p_hidden=float(cfg.get("gru4rec_dropout_hidden", 0.05)),
            learning_rate=float(cfg.get("gru4rec_lr", 0.05)), momentum=float(cfg.get("gru4rec_momentum", 0.4)),
            sample_alpha=float(cfg.get("gru4rec_sample_alpha", 0.4)), n_sample=int(cfg.get("gru4rec_n_sample", 2048)),
            embedding=0, constrained_embedding=True, n_epochs=int(cfg.get("gru4rec_epochs", 10)),
            bpreg=float(cfg.get("gru4rec_bpreg", 1.95)), elu_param=0.5 if loss == "bpr-max" else 0.0,
            logq=float(cfg.get("gru4rec_logq", 0.0)), device=self.device,
        )
        with contextlib.redirect_stdout(io.StringIO()) as log:  # the official code prints every epoch
            # compatibility_mode=False: the code's own PyTorch initialisation. Its Theano-compatible mode builds
            # float64 weights under NumPy 2's type promotion and fails.
            model.fit(frame, sample_cache_max_size=int(cfg.get("gru4rec_sample_cache", 10_000_000)), compatibility_mode=False)
        if getattr(model, "error_during_train", False):
            raise RuntimeError("GRU4Rec training diverged (NaN loss); try a lower learning rate")
        self.model = model.model
        self.model.eval()
        itemidmap = model.data_iterator.itemidmap
        self.to_model = np.full(self.n_items + 1, -1, dtype=np.int64)
        self.to_model[itemidmap.index.to_numpy(dtype=np.int64)] = itemidmap.to_numpy(dtype=np.int64)
        self.from_model = itemidmap.index.to_numpy(dtype=np.int64)
        self.fit_info = {"epochs": int(cfg.get("gru4rec_epochs", 10)), "log_tail": log.getvalue().strip().splitlines()[-1][:200] if log.getvalue() else ""}

    def score_users(self, users: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        import torch

        items = self.to_model[hist.items[:, -self.seq_len :]]  # -1 for padding or items unknown to the model
        model = self.model
        with torch.no_grad():
            hidden = [torch.zeros((len(users), size), device=self.device) for size in model.layers]
            for t in range(items.shape[1]):
                step = torch.as_tensor(items[:, t], device=self.device)
                valid = step >= 0
                if not bool(valid.any()):
                    continue
                x = model.Wy(step.clamp(min=0))  # constrained embedding: inputs share the output item embeddings
                for layer, cell in enumerate(model.G):
                    new = cell(x, hidden[layer])
                    hidden[layer] = torch.where(valid[:, None], new, hidden[layer])  # padding leaves the state unchanged
                    x = hidden[layer]
            scores = model.score_items(hidden[-1], model.Wy.weight, model.By.weight).float().cpu().numpy()
        out = np.full((len(users), self.n_items + 1), NEG_INF, dtype=np.float32)
        out[:, self.from_model] = scores
        return out
