"""Train RecBole BERT4Rec/S3Rec/DIN on our train split and score the candidate file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

# RecBole 1.2 restores NumPy 1 aliases that NumPy 2 removed.
for _name, _value in {
    "bool_": bool,
    "int_": int,
    "float_": float,
    "complex_": complex,
    "object_": object,
    "str_": str,
    "unicode_": str,
}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _value)

import pandas as pd
import pyarrow.parquet as pq
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.utils import get_model, init_seed

from recbench.methods._common import TrainConfig, nhead
from recbench.protocol import Explanation, Recommender


def _token(value: object) -> str:
    return str(value).replace("\t", " ").replace("\n", " ").replace('"', "").replace("\\", "")


def _export_atomic(store: Any, folder: Path, name: str, with_item_feature: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    train = pq.read_table(
        store.train_path, columns=["user_id", "item_id", "timestamp"]
    ).to_pandas()
    train["timestamp"] = pd.to_datetime(train["timestamp"]).astype("int64") / 1e9
    inter = folder / f"{name}.inter"
    lines = ["user_id:token\titem_id:token\ttimestamp:float"]
    for user, item, stamp in zip(train["user_id"], train["item_id"], train["timestamp"]):
        lines.append(f"{_token(user)}\t{_token(item)}\t{float(stamp):.0f}")
    inter.write_text("\n".join(lines) + "\n")
    if with_item_feature:
        items = pq.read_table(store.items_path, columns=["item_id", "category"]).to_pandas()
        item_file = folder / f"{name}.item"
        rows = ["item_id:token\tcategory:token"]
        for item_id, category in zip(items["item_id"], items["category"]):
            token = _token(category or "unk").replace(" ", "_") or "unk"
            rows.append(f"{_token(item_id)}\t{token}")
        item_file.write_text("\n".join(rows) + "\n")


class RecBoleMethod(Recommender):
    model_name = "BERT4Rec"
    needs_item_feature = False

    def fit(self, store: Any, config: dict[str, Any]) -> None:
        cfg = TrainConfig.from_dict(config)
        init_seed(42, True)
        work = Path(store.root) / "recbole" / self.spec.name
        dataset_name = "rbset"
        atomic = work / dataset_name
        _export_atomic(store, atomic, dataset_name, self.needs_item_feature)
        hidden = max(cfg.dim, 16)
        heads = nhead(hidden)
        config_dict = {
            "load_col": (
                {"inter": ["user_id", "item_id", "timestamp"], "item": ["item_id", "category"]}
                if self.needs_item_feature
                else {"inter": ["user_id", "item_id", "timestamp"]}
            ),
            "TIME_FIELD": "timestamp",
            "data_path": str(work),
            "checkpoint_dir": str(work / "ckpt"),
            "log_dir": str(work / "log"),
            "epochs": 1,
            "train_batch_size": cfg.batch_size,
            "eval_batch_size": cfg.batch_size,
            "learning_rate": cfg.lr,
            "embedding_size": hidden,
            "hidden_size": hidden,
            "inner_size": hidden * 2,
            "n_layers": max(cfg.layers, 1),
            "n_heads": heads,
            "hidden_dropout_prob": 0.0,
            "attn_dropout_prob": 0.0,
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-12,
            "initializer_range": 0.02,
            "mask_ratio": 0.2,
            "loss_type": "BPR",
            "train_neg_sample_args": {
                "distribution": "uniform",
                "sample_num": 1,
                "dynamic": False,
                "candidate_num": 0,
            },
            "MAX_ITEM_LIST_LENGTH": cfg.seq_len,
            "metrics": ["Recall"],
            "topk": [10],
            "valid_metric": "Recall@10",
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "order": "TO",
                "group_by": "user",
                "mode": "full",
            },
            "train_stage": "pretrain",
            "pre_model_path": "",
            "item_attribute": "category",
            "aap_weight": 0.2,
            "mip_weight": 1.0,
            "map_weight": 1.0,
            "sp_weight": 0.5,
            "mlp_hidden_size": [hidden, hidden],
            "dropout_prob": 0.0,
            "pooling_mode": "mean",
            "numerical_features": [],
            "show_progress": False,
            "device": cfg.device,
            "use_gpu": cfg.device == "cuda",
            "reproducibility": True,
            "seed": 42,
            "state": "error",
            "save_dataset": False,
            "save_dataloaders": False,
            "stopping_step": 100,
            "eval_step": 100,
        }
        (work / "ckpt").mkdir(parents=True, exist_ok=True)
        (work / "log").mkdir(parents=True, exist_ok=True)
        previous = Path.cwd()
        os.chdir(work)
        try:
            rec_config = Config(model=self.model_name, dataset=dataset_name, config_dict=config_dict)
            dataset = create_dataset(rec_config)
            train_data, _valid, _test = data_preparation(rec_config, dataset)
            model = get_model(self.model_name)(rec_config, train_data.dataset).to(rec_config["device"])
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
            steps = 0
            while steps < cfg.max_steps:
                progressed = False
                for batch in train_data:
                    progressed = True
                    interaction = batch[0] if isinstance(batch, tuple) else batch
                    interaction = interaction.to(rec_config["device"])
                    optimizer.zero_grad()
                    loss = model.calculate_loss(interaction)
                    if isinstance(loss, tuple):
                        loss = loss[0]
                    loss.backward()
                    optimizer.step()
                    steps += 1
                    if steps >= cfg.max_steps:
                        break
                if not progressed or steps >= cfg.max_steps:
                    break
        finally:
            os.chdir(previous)
        model.eval()
        self.model = model
        self.dataset = train_data.dataset
        self.rec_config = rec_config
        self.cfg = cfg
        self._bind(store)

    def _bind(self, store: Any) -> None:
        user_to, idx_user, item_to, idx_item = store.id_maps()
        self.user_to = user_to
        self.idx_user = idx_user
        self.item_to = item_to
        self.idx_item = idx_item
        self.store = store

    def _token_id(self, field: str, token: str) -> int | None:
        table = self.dataset.field2token_id.get(field, {})
        return table.get(str(token))

    def score_candidates(self, store: Any, candidates: Any = None):
        if candidates is None:
            candidates = pq.read_table(store.candidates_path).to_pandas()
        self._bind(store)
        frame = candidates.copy()
        scores = np.full(len(frame), -1e9, dtype=np.float32)
        hist_map = _histories(store)
        seq_len = int(self.rec_config["MAX_ITEM_LIST_LENGTH"])
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        users = frame["user_idx"].to_numpy()
        items = frame["item_idx"].to_numpy()
        self.model.eval()
        device = self.rec_config["device"]
        with torch.no_grad():
            for start in range(0, len(frame), 256):
                stop = min(start + 256, len(frame))
                seqs = []
                lengths = []
                targets = []
                keep = []
                for row, user_idx, item_idx in zip(range(start, stop), users[start:stop], items[start:stop]):
                    user_token = self.idx_user.get(int(user_idx))
                    item_token = self.idx_item.get(int(item_idx))
                    if user_token is None or item_token is None:
                        continue
                    user_id = self._token_id(uid_field, user_token)
                    item_id = self._token_id(iid_field, item_token)
                    if user_id is None or item_id is None:
                        continue
                    raw_hist = hist_map.get(int(user_idx), [])
                    mapped = []
                    for raw in raw_hist[-seq_len:]:
                        token = self.idx_item.get(int(raw))
                        if token is None:
                            continue
                        mapped_id = self._token_id(iid_field, token)
                        if mapped_id:
                            mapped.append(mapped_id)
                    if not mapped:
                        mapped = [0]
                    pad = [0] * (seq_len - len(mapped))
                    seqs.append(pad + mapped[-seq_len:])
                    lengths.append(max(len(mapped), 1))
                    targets.append(item_id)
                    keep.append(row)
                if not keep:
                    continue
                target_tensor = torch.tensor(targets, dtype=torch.long, device=device)
                fields = {
                    self.model.ITEM_SEQ: torch.tensor(seqs, dtype=torch.long, device=device),
                    self.model.ITEM_SEQ_LEN: torch.tensor(lengths, dtype=torch.long, device=device),
                    self.model.ITEM_ID: target_tensor,
                    self.model.POS_ITEM_ID: target_tensor,
                    uid_field: torch.tensor(
                        [self._token_id(uid_field, self.idx_user[int(users[i])]) or 0 for i in keep],
                        dtype=torch.long,
                        device=device,
                    ),
                }
                interaction = Interaction(fields)
                batch_scores = self.model.predict(interaction).detach().float().cpu().numpy()
                scores[keep] = batch_scores
        frame["score"] = scores
        return frame

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.model.state_dict(), "model_name": self.model_name}, path)

    def load(self, path: str) -> None:
        raise RuntimeError("Reload a RecBole checkpoint by fitting again in this smoke build.")

    def recommend(self, user_id: str, k: int = 10) -> list[dict[str, Any]]:
        raise RuntimeError("Use score_candidates for RecBole methods.")

    def explain_global(self) -> Explanation | None:
        return Explanation("global", self.spec.upstream, [])

    def explain_local(self, user_id: str, item_ids: list[str]) -> list[Explanation]:
        return [
            Explanation("local", f"{self.spec.name} scores {item_id} from the cutoff-truncated history of {user_id}.", [{"item_id": item_id}])
            for item_id in item_ids
        ]

    def _bind_called(self) -> None:
        return None


def _histories(store: Any) -> dict[int, list[int]]:
    table = pq.read_table(store.sequences_path, columns=["user_idx", "history"]).to_pandas()
    histories = {}
    for user, hist in zip(table["user_idx"], table["history"]):
        values = [] if hist is None else [int(item) for item in list(hist)]
        histories[int(user)] = values
    return histories
