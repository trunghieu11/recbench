"""BERT4Rec, S3-Rec, and DIN trained through RecBole 1.2 on the pre-test data only.

How the adapter works:
1. Export every pre-test event as RecBole "atomic" files. Events are numbered in
   (user, time, original order) so ties never reorder; categories become a
   token_seq item feature.
2. Build RecBole's dataset with split ratios [1.0, 0, 0] so RecBole trains on
   100% of what it is given (we do our own train/test separation upstream).
3. Train with RecBole's own data loader (it creates BERT4Rec's masks and DIN's
   negatives) for a fixed number of steps.
4. Score with RecBole's layout: items first, padding at the end
   (HistoryBatch.left_aligned()). Ids are mapped between recbench and RecBole.

Model notes: BERT4Rec and S3-Rec use full softmax cross-entropy (stronger than
BPR, see Petrov & Macdonald 2022); S3-Rec is pre-trained with its
self-supervised objectives and then fine-tuned, as in the paper; DIN is a
pointwise click model (sigmoid output) and scores (user, item) pairs.
"""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

# RecBole 1.2 still uses NumPy 1 aliases that NumPy 2 removed.
for _name, _value in {"bool_": bool, "int_": int, "float_": float, "complex_": complex, "object_": object, "str_": str, "unicode_": str}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _value)

import torch  # noqa: E402

# RecBole 1.2 calls pandas in ways pandas 3 will change; the warnings are noise for us.
warnings.filterwarnings("ignore", category=FutureWarning, module="recbole")
warnings.filterwarnings("ignore", message=".*inplace method.*")

from recbench.data import HistoryBatch, TrainView  # noqa: E402
from recbench.methods._explain import embedding_explanations  # noqa: E402
from recbench.methods._torch import resolve_device  # noqa: E402
from recbench.methods.sasrec import SEQ_TASKS  # noqa: E402
from recbench.protocol import NEG_INF, MethodSpec, Recommender, Task, Unsupported  # noqa: E402
from recbench.registry import register_method  # noqa: E402


@contextmanager
def _inside(folder: Path):
    previous = Path.cwd()
    folder.mkdir(parents=True, exist_ok=True)
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(previous)


def _clean_token(value: object) -> str:
    return str(value).replace("\t", " ").replace("\n", " ").replace('"', "").replace("\\", "").strip()


class RecBoleMethod(Recommender):
    model_name = "BERT4Rec"
    uses_categories = False

    # ----- export -----
    def _export(self, data: TrainView, folder: Path) -> None:
        frame = data.export_frame()
        folder.mkdir(parents=True, exist_ok=True)
        order = np.arange(len(frame), dtype=np.int64)  # strictly increasing "time" in (user, ts, event_seq) order
        with (folder / "rb.inter").open("w") as out:
            out.write("user_id:token\titem_id:token\ttimestamp:float\n")
            for user, item, stamp in zip(frame["user_id"].map(_clean_token), frame["item_id"].map(_clean_token), order):
                out.write(f"{user}\t{item}\t{stamp}\n")
        if self.uses_categories:
            with (folder / "rb.item").open("w") as out:
                out.write("item_id:token\tcategory:token_seq\n")
                for item_id, category in zip(data.item_ids[1:], data.item_category[1:]):
                    tokens = " ".join(_clean_token(t).replace(" ", "_") for t in str(category).split("|") if t.strip()) or "unk"
                    out.write(f"{_clean_token(item_id)}\t{tokens}\n")

    def _config(self, data: TrainView, cfg: dict[str, Any], work: Path, extra: dict[str, Any]) -> dict[str, Any]:
        dim = max(int(cfg.get("dim", 64)), 16)
        load_col = {"inter": ["user_id", "item_id", "timestamp"]}
        if self.uses_categories:
            load_col["item"] = ["item_id", "category"]
        device = resolve_device(cfg)
        base = {
            "load_col": load_col,
            "TIME_FIELD": "timestamp",
            "data_path": str(work),
            "checkpoint_dir": str(work / "ckpt"),
            "train_batch_size": int(cfg.get("batch_size", 128)),
            "eval_batch_size": int(cfg.get("batch_size", 128)),
            "learning_rate": float(cfg.get("lr", 1e-3)),
            "embedding_size": dim,
            "hidden_size": dim,
            "inner_size": 2 * dim,
            "n_layers": int(cfg.get("layers", 2)),
            "n_heads": int(cfg.get("heads", 2)),
            "hidden_dropout_prob": float(cfg.get("dropout", 0.2)),
            "attn_dropout_prob": float(cfg.get("dropout", 0.2)),
            "MAX_ITEM_LIST_LENGTH": int(cfg.get("seq_len", 50)),
            "eval_args": {"split": {"RS": [1.0, 0.0, 0.0]}, "order": "TO", "group_by": "user", "mode": "full"},
            "metrics": ["Recall"],
            "topk": [10],
            "valid_metric": "Recall@10",
            "device": str(device),
            "use_gpu": device.type == "cuda",
            "seed": int(cfg.get("seed", 42)),
            "reproducibility": True,
            "show_progress": False,
            "state": "ERROR",
            "save_dataset": False,
            "save_dataloaders": False,
            "user_inter_num_interval": "[0,inf)",
            "item_inter_num_interval": "[0,inf)",
        }
        return {**base, **extra}

    def _memory_check(self, data: TrainView, cfg: dict[str, Any]) -> None:
        import psutil

        seq_len = int(cfg.get("seq_len", 50))
        needed = data.meta.get("n_pretest", 0) * seq_len * 16  # item list + timestamp list per event
        available = psutil.virtual_memory().available
        if needed > 0.6 * available:
            raise Unsupported(
                f"RecBole's sequence dataset would need ~{needed / 2**30:.1f} GB (available {available / 2**30:.1f} GB); "
                "lower seq_len or use SASRec"
            )

    def _build(self, data: TrainView, cfg: dict[str, Any], extra: dict[str, Any], work: Path):
        from recbole.config import Config
        from recbole.data import create_dataset
        from recbole.data.dataloader import TrainDataLoader
        from recbole.data.utils import create_samplers
        from recbole.utils import get_model, init_seed

        config = Config(model=self.model_name, dataset="rb", config_dict=self._config(data, cfg, work, extra))
        init_seed(config["seed"], config["reproducibility"])
        dataset = create_dataset(config)
        built = dataset.build()
        train_sampler, _, _ = create_samplers(config, dataset, built)
        loader = TrainDataLoader(config, built[0], train_sampler, shuffle=True)
        model = get_model(self.model_name)(config, built[0]).to(config["device"])
        return config, built[0], loader, model

    def _train(self, model, loader, steps: int, lr: float, device) -> float:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        done, last = 0, 0.0
        while done < steps:
            progressed = False
            for batch in loader:
                progressed = True
                interaction = (batch[0] if isinstance(batch, tuple) else batch).to(device)
                loss = model.calculate_loss(interaction)
                loss = loss[0] if isinstance(loss, tuple) else loss
                if isinstance(loss, (list, tuple)):
                    loss = sum(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last = float(loss.detach())
                done += 1
                if done >= steps:
                    break
            if not progressed:
                break
        model.eval()
        return last

    # ----- fit / score -----
    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self.bind(data)
        self._memory_check(data, cfg)
        self.seq_len = int(cfg.get("seq_len", 50))
        work = data.cache_dir / "recbole" / self.spec.name
        self._export(data, work / "rb")
        with _inside(work):
            self.config, self.rb_data, loader, self.model = self._build(data, cfg, self.model_extra(cfg), work)
            last = self._train(self.model, loader, int(cfg.get("max_steps", 400)), float(cfg.get("lr", 1e-3)), self.config["device"])
        self.fit_info = {"last_loss": last}
        self._map_ids(data)

    def model_extra(self, cfg: dict[str, Any]) -> dict[str, Any]:
        return {}

    def _map_ids(self, data: TrainView) -> None:
        dataset = self.rb_data
        item_tokens = dataset.field2token_id[dataset.iid_field]
        user_tokens = dataset.field2token_id[dataset.uid_field]
        self.rb_item = np.array([item_tokens.get(_clean_token(i), -1) for i in data.item_ids], dtype=np.int64)
        self.rb_item[0] = 0
        self.rb_user = np.array([user_tokens.get(_clean_token(u), 0) for u in data.user_ids], dtype=np.int64)
        self.device = self.config["device"]

    def _sequences(self, hist: HistoryBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Our right-aligned histories -> RecBole ids, left-aligned, exactly MAX_ITEM_LIST_LENGTH wide."""
        mapped = self.rb_item[hist.items[:, -self.seq_len :]]
        mapped = np.where(mapped > 0, mapped, 0)
        width = mapped.shape[1]
        lengths = (mapped > 0).sum(axis=1)
        # Drop unknown ids, then left-align (stable: keeps the time order).
        order = np.argsort(mapped == 0, axis=1, kind="stable")
        left = np.take_along_axis(mapped, order, axis=1)
        if width < self.seq_len:
            left = np.pad(left, ((0, 0), (0, self.seq_len - width)))
        to_t = lambda a: torch.as_tensor(a, dtype=torch.long, device=self.device)  # noqa: E731
        return to_t(left), to_t(np.maximum(lengths, 1))

    def _to_ours(self, rb_scores: np.ndarray) -> np.ndarray:
        out = np.full((rb_scores.shape[0], self.n_items + 1), NEG_INF, dtype=np.float32)
        known = self.rb_item > 0
        out[:, known] = rb_scores[:, self.rb_item[known]]
        out[:, 0] = NEG_INF
        return out

    def item_embeddings(self) -> np.ndarray | None:
        weight = getattr(self.model, "item_embedding", None)
        if weight is None:
            return None
        rb = weight.weight.detach().float().cpu().numpy()
        out = np.zeros((self.n_items + 1, rb.shape[1]), dtype=np.float32)
        known = self.rb_item > 0
        out[known] = rb[self.rb_item[known]]
        return out

    def explain(self, users, items, hist):
        vecs = self.item_embeddings()
        if vecs is None:
            return [[] for _ in users]
        return embedding_explanations(vecs, users, items, hist, self.item_ids, self.spec.name)


class _FullSortMethod(RecBoleMethod):
    @torch.no_grad()
    def score_users(self, users: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        from recbole.data.interaction import Interaction

        seq, lengths = self._sequences(hist)
        interaction = Interaction({self.model.ITEM_SEQ: seq, self.model.ITEM_SEQ_LEN: lengths}).to(self.device)
        return self._to_ours(self.model.full_sort_predict(interaction).float().cpu().numpy())


@register_method
class BERT4Rec(_FullSortMethod):
    model_name = "BERT4Rec"
    spec = MethodSpec(
        name="bert4rec",
        tasks=SEQ_TASKS,
        uses_history=True,
        needs_torch=True,
        upstream="RecBole 1.2 BERT4Rec (CE loss, mask_ratio 0.2)",
        cost_band="medium",
    )

    def model_extra(self, cfg):
        return {"loss_type": "CE", "train_neg_sample_args": None, "mask_ratio": float(cfg.get("mask_ratio", 0.2)), "transform": "mask_itemseq"}


@register_method
class S3Rec(_FullSortMethod):
    model_name = "S3Rec"
    uses_categories = True
    spec = MethodSpec(
        name="s3rec",
        tasks=SEQ_TASKS,
        uses_history=True,
        requires_side_features=True,
        needs_torch=True,
        upstream="RecBole 1.2 S3Rec: pretrain (AAP/MIP/MAP/SP) then finetune (CE)",
        cost_band="medium",
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        tokens = {t for c in data.item_category[1:] for t in str(c).split("|") if t}
        if len(tokens) < 2:
            raise Unsupported("S3-Rec needs item attributes; this dataset has fewer than 2 category values")
        self.bind(data)
        self._memory_check(data, cfg)
        self.seq_len = int(cfg.get("seq_len", 50))
        work = data.cache_dir / "recbole" / self.spec.name
        self._export(data, work / "rb")
        steps = int(cfg.get("max_steps", 400))
        common = {"item_attribute": "category", "train_neg_sample_args": None, "mask_ratio": 0.2,
                  "aap_weight": 0.2, "mip_weight": 1.0, "map_weight": 1.0, "sp_weight": 0.5}
        pretrained = work / "pretrained.pth"
        with _inside(work):
            config, _, loader, model = self._build(data, cfg, {**common, "train_stage": "pretrain", "pre_model_path": "", "loss_type": "CE"}, work)
            pre_loss = self._train(model, loader, steps // 2, float(cfg.get("lr", 1e-3)), config["device"])
            torch.save({"state_dict": model.state_dict()}, pretrained)
            self.config, self.rb_data, loader, self.model = self._build(
                data, cfg, {**common, "train_stage": "finetune", "pre_model_path": str(pretrained), "loss_type": "CE"}, work
            )
            fine_loss = self._train(self.model, loader, steps - steps // 2, float(cfg.get("lr", 1e-3)), self.config["device"])
        self.fit_info = {"pretrain_last_loss": pre_loss, "finetune_last_loss": fine_loss}
        self._map_ids(data)


@register_method
class DIN(RecBoleMethod):
    model_name = "DIN"
    uses_categories = True
    pair_budget = 1 << 15  # each pair carries the whole history through attention
    spec = MethodSpec(
        name="din",
        tasks={Task.topn, Task.ctr, Task.sequential},
        output="pairs",
        uses_history=True,
        outputs_probability=True,
        needs_torch=True,
        upstream="RecBole 1.2 DIN (attention over the history, BCE with sampled negatives)",
        cost_band="high",
    )

    def model_extra(self, cfg):
        dim = max(int(cfg.get("dim", 64)), 16)
        return {
            "train_neg_sample_args": {"distribution": "uniform", "sample_num": int(cfg.get("din_negatives", 4)), "dynamic": False, "candidate_num": 0},
            "mlp_hidden_size": [2 * dim, dim],
            "dropout_prob": 0.0,
            "pooling_mode": "mean",
            "numerical_features": [],
        }

    @torch.no_grad()
    def score_pairs(self, users: np.ndarray, items: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        seq, lengths = self._sequences(hist)
        n_cand = items.shape[1]
        flat_items = self.rb_item[items.ravel()]
        known = flat_items > 0
        rb_users = torch.as_tensor(self.rb_user[np.repeat(users, n_cand)], dtype=torch.long, device=self.device)
        logits = self.model.forward(
            rb_users,
            seq.repeat_interleave(n_cand, dim=0),
            lengths.repeat_interleave(n_cand, dim=0),
            torch.as_tensor(np.where(known, flat_items, 1), dtype=torch.long, device=self.device),
        )
        out = logits.float().cpu().numpy()
        out[~known] = NEG_INF
        return out.reshape(len(users), n_cand)
