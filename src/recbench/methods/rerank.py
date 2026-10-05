"""Two-stage recommendation: cheap candidate generators, then a learned re-ranker (LightGBM or DCN-V2).

Stage 1, retrieval: EASE, ItemKNN and recent popularity each propose items, and their union (up to
`rerank_candidates`, default 200) becomes the candidate list. The generators use the settings the bake-off
chose for them on this dataset (runs/tuning/<tier>/<dataset>/<method>.json, or its confirmation on full data),
or their defaults. A tuning job pins them in the re-ranker's config, so they are part of every run's identity.

Stage 2, ranking: a model scores every (user, candidate) pair from features. These are the generators'
scores and ranks, item popularity and trend, item age, user activity, category affinity, co-visitation with
the user's last item, and (optionally) text similarity.

Training without leakage: the ranker learns from the most recent part of the pre-test data. Generators are
fitted on the events before this split's validation cutoff, and candidates and features are computed as of
that cutoff. The labels are the candidates each user actually chose between that cutoff and the test cutoff.
To score, the generators are refitted on all pre-test data and features are computed as of the test cutoff.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import scipy.sparse as sp

from recbench.data import DAY_US, HistoryBatch, TrainView
from recbench.methods.baselines import EASE, ItemKNN, MostPopular
from recbench.protocol import NEG_INF, Explanation, MethodSpec, Recommender, Task, Unsupported
from recbench.registry import register_method

FEATURES = (
    "ease_score", "ease_rank", "knn_score", "knn_rank", "pop_rank", "best_rank", "n_sources",
    "item_pop", "item_pop_7d", "item_pop_28d", "item_trend", "item_age_days",
    "user_events", "user_items", "user_days_idle", "category_share", "covisit_last", "text_similarity",
)


def tuned_params(data: TrainView, method: str) -> dict[str, Any]:
    """Best settings found for `method` on this dataset: confirmed on this tier, else tuned on this tier, else
    tuned on the quick tier ({} when not tuned yet). Tuning jobs pin them in the config (`rerank_ease`,
    `rerank_itemknn`) when they start, scaled to the data size, so this lookup only fills what a job did not pin
    (for example a direct fit in a notebook)."""
    from recbench.tuning.job import generator_settings

    return generator_settings(data.tier, data.dataset, method)


def text_vectors(data: TrainView, model_name: str, device: str) -> np.ndarray:
    """The items' text vectors from the split's cache. A missing cache is filled by a separate process: the sentence
    encoder needs PyTorch, and on macOS PyTorch's OpenMP runtime crashes LightGBM's when both are in one process."""
    from recbench.methods.text_knn import vectors_path

    path = vectors_path(data, model_name)
    if not path.exists():
        import os
        import subprocess

        script = ("import sys\nfrom recbench.data import TrainView\nfrom recbench.methods._torch import resolve_device\n"
                  "from recbench.methods.text_knn import cached_item_vectors\n"
                  "cached_item_vectors(TrainView(sys.argv[1]), sys.argv[2], resolve_device({'device': sys.argv[3]}).type)\n")
        subprocess.run([sys.executable, "-c", script, str(data.root), model_name, device], check=True,
                       env={**os.environ, "RECBENCH_METHOD_MODULES": "text_knn"})
    return np.load(path)


class Generators:
    """EASE + ItemKNN + recent popularity, fitted on one view; proposes candidates with their scores and ranks."""

    def __init__(self, cfg: dict[str, Any], ease_cfg: dict[str, Any], knn_cfg: dict[str, Any]):
        self.cfg, self.ease_cfg, self.knn_cfg = cfg, ease_cfg, knn_cfg
        self.n_candidates = int(cfg.get("rerank_candidates", 200))

    def fit(self, view: TrainView) -> "Generators":
        self.view = view
        self.ease, self.knn, self.pop = EASE(), ItemKNN(), MostPopular()
        self.ease.fit(view, {**self.cfg, **self.ease_cfg})
        self.knn.fit(view, {**self.cfg, **self.knn_cfg})
        self.pop.fit(view, {**self.cfg, "pop_window_days": int(self.cfg.get("rerank_pop_days", 7))})
        self.pop_order = np.argsort(-self.pop.scores, kind="stable")
        return self

    def candidates(self, users: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """[B, C] candidate items (0 = empty slot) and per-candidate generator features."""
        hist = self.view.history_batch(users, 1)
        seen = self.view.seen[users]
        rows, cols = seen.nonzero()
        k = self.n_candidates
        lists, scores = {}, {}
        for name, model, take in (("ease", self.ease, k * 3 // 4), ("knn", self.knn, k // 2)):
            s = model.score_users(users, hist)
            s[:, 0] = NEG_INF
            s[rows, cols] = NEG_INF
            top = np.argpartition(-s, min(take, s.shape[1] - 1) - 1, axis=1)[:, :take]
            order = np.take_along_axis(top, np.argsort(-np.take_along_axis(s, top, 1), axis=1, kind="stable"), 1)
            lists[name], scores[name] = order, s
        popular = self.pop_order[self.pop_order > 0][: k // 4 + 50]
        cands = np.zeros((len(users), k), dtype=np.int64)
        feats = {f: np.zeros((len(users), k), dtype=np.float32) for f in ("ease_score", "ease_rank", "knn_score", "knn_rank", "pop_rank", "best_rank", "n_sources")}
        pop_rank = np.full(self.view.n_items + 1, 1e4, dtype=np.float32)
        pop_rank[popular] = np.arange(1, len(popular) + 1)
        for b in range(len(users)):
            seen_b = set(seen[b].indices.tolist())
            ranks: dict[int, list[float]] = {}
            proposals = (
                ("ease", [i for i in lists["ease"][b] if np.isfinite(scores["ease"][b, i])]),  # seen items score -inf
                ("knn", [i for i in lists["knn"][b] if np.isfinite(scores["knn"][b, i])]),
                ("pop", [i for i in popular if i not in seen_b][: k // 4]),
            )
            for source, items in proposals:
                for rank, item in enumerate(items, start=1):
                    entry = ranks.setdefault(int(item), [1e4, 1e4, 1e4])
                    entry[("ease", "knn", "pop").index(source)] = rank
            chosen = sorted(ranks, key=lambda i: min(ranks[i]))[:k]
            cands[b, : len(chosen)] = chosen
            for c, item in enumerate(chosen):
                r = ranks[item]
                feats["ease_rank"][b, c], feats["knn_rank"][b, c], feats["pop_rank"][b, c] = r
                feats["best_rank"][b, c] = min(r)
                feats["n_sources"][b, c] = sum(x < 1e4 for x in r)
            feats["ease_score"][b, : len(chosen)] = scores["ease"][b, chosen]
            feats["knn_score"][b, : len(chosen)] = scores["knn"][b, chosen]
        for name in ("ease_score", "knn_score"):
            feats[name] = np.where(np.isfinite(feats[name]), feats[name], 0.0).astype(np.float32)
        return cands, feats


class FeatureBuilder:
    """Item, user and pair features as of one view's cutoff (no event at or after it is ever used)."""

    def __init__(self, view: TrainView, text_vectors: np.ndarray | None = None):
        self.view = view
        events = view.events()
        cutoff = int(view.meta["test_start_us"])
        age = (cutoff - events["ts_us"].to_numpy()) / DAY_US
        items = events["item_idx"].to_numpy()
        n = view.n_items + 1
        self.pop = view.item_pop.astype(np.float32)
        self.pop_7 = np.bincount(items[age <= 7], minlength=n).astype(np.float32)
        self.pop_28 = np.bincount(items[age <= 28], minlength=n).astype(np.float32)
        first = np.full(n, np.inf)
        np.minimum.at(first, items, events["ts_us"].to_numpy().astype(np.float64))
        self.age = np.where(np.isfinite(first), (cutoff - first) / DAY_US, 0.0).astype(np.float32)
        lengths = view.user_lengths
        offsets = np.asarray(view._offsets)
        ts = np.asarray(view._ts)
        last = np.where(lengths > 0, ts[np.maximum(offsets[1:] - 1, 0)], cutoff)
        self.user_events = lengths.astype(np.float32)
        self.user_items = np.diff(view.seen.indptr).astype(np.float32)
        self.user_idle = ((cutoff - last) / DAY_US).astype(np.float32)
        flat = np.asarray(view._items)
        self.last_item = np.where(lengths > 0, flat[np.maximum(offsets[1:] - 1, 0)], 0)
        owner = np.repeat(np.arange(len(lengths)), lengths)
        same_user = owner[1:] == owner[:-1]
        self.covisit = sp.csr_matrix((np.ones(int(same_user.sum()), dtype=np.float32), (flat[:-1][same_user], flat[1:][same_user])), shape=(n, n))
        self.covisit.sum_duplicates()
        cats = [str(c).split("|")[0] for c in view.item_category]
        vocab = {c: i for i, c in enumerate(sorted(set(cats)))}
        self.item_cat = np.array([vocab[c] for c in cats], dtype=np.int64)
        self.n_cats = len(vocab)
        self.text = text_vectors

    def build(self, users: np.ndarray, cands: np.ndarray, gen_feats: dict[str, np.ndarray]) -> np.ndarray:
        """[B * C, len(FEATURES)] rows in the order of cands.ravel()."""
        view = self.view
        flat = cands.ravel()
        b_users = np.repeat(users, cands.shape[1])
        cat_counts = sp.csr_matrix((np.ones(view.seen[users].nnz, dtype=np.float32),
                                    (np.repeat(np.arange(len(users)), np.diff(view.seen[users].indptr)), self.item_cat[view.seen[users].indices])),
                                   shape=(len(users), self.n_cats)).toarray()
        share = cat_counts / np.maximum(cat_counts.sum(axis=1, keepdims=True), 1.0)
        category_share = share[np.repeat(np.arange(len(users)), cands.shape[1]), self.item_cat[flat]]
        covisit = np.asarray(self.covisit[self.last_item[b_users], flat]).ravel()
        if self.text is not None:
            recent = view.history_batch(users, 20).items
            profile = self.text[recent].sum(axis=1)
            profile /= np.maximum(np.linalg.norm(profile, axis=1, keepdims=True), 1e-9)
            text_sim = np.einsum("bd,bcd->bc", profile, self.text[cands]).ravel()
        else:
            text_sim = np.zeros(len(flat), dtype=np.float32)
        columns = {
            **{k: v.ravel() for k, v in gen_feats.items()},
            "item_pop": self.pop[flat], "item_pop_7d": self.pop_7[flat], "item_pop_28d": self.pop_28[flat],
            "item_trend": self.pop_7[flat] / (self.pop_28[flat] + 1.0), "item_age_days": self.age[flat],
            "user_events": self.user_events[b_users], "user_items": self.user_items[b_users], "user_days_idle": self.user_idle[b_users],
            "category_share": category_share, "covisit_last": covisit, "text_similarity": text_sim,
        }
        return np.stack([np.asarray(columns[name], dtype=np.float32) for name in FEATURES], axis=1)


class TwoStage(Recommender):
    """Shared training-table construction and scoring; subclasses fit and apply the ranking model."""

    def _setup(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self.bind(data)
        self.data, self.cfg = data, cfg
        self.ease_cfg = {**tuned_params(data, "ease"), **(cfg.get("rerank_ease") or {})}
        self.knn_cfg = {**tuned_params(data, "itemknn"), **(cfg.get("rerank_itemknn") or {})}
        text = None
        if cfg.get("rerank_text") and (any(data.item_text[1:]) or any(data.item_category[1:])):
            from recbench.methods.text_knn import DEFAULT_ENCODER

            text = text_vectors(data, str(cfg.get("text_encoder") or DEFAULT_ENCODER), str(cfg.get("device", "auto")))
        self.text = text

    def training_table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(features, labels, user index per row, candidate item per row) from the window before the test cutoff."""
        data, cfg = self.data, self.cfg
        cutoff = int(data.meta.get("valid_start_us") or 0)
        if not cutoff or cutoff >= int(data.meta["test_start_us"]):
            raise Unsupported("this split has no validation window to train a re-ranker on")
        past = data.before(cutoff)
        events = data.events()
        window = events[events["ts_us"] >= cutoff]
        new = window[np.asarray(past.seen[window["user_idx"].to_numpy(), window["item_idx"].to_numpy()]).ravel() == 0]
        targets = new.groupby("user_idx")["item_idx"].apply(lambda s: set(s.tolist()))
        users = np.array([u for u in targets.index if past.user_lengths[u] > 0], dtype=np.int64)
        rng = np.random.default_rng(int(cfg.get("seed", 42)))
        cap = int(cfg.get("rerank_train_users", 20_000))
        if len(users) > cap:
            users = np.sort(rng.permutation(users)[:cap])
        gens = Generators(cfg, self.ease_cfg, self.knn_cfg).fit(past)
        builder = FeatureBuilder(past, self.text)
        x_parts, y_parts, u_parts, i_parts = [], [], [], []
        for start in range(0, len(users), 512):  # each batch holds two users x catalog score matrices
            batch = users[start : start + 512]
            cands, feats = gens.candidates(batch)
            labels = np.array([[int(item in targets[u]) for item in row] for u, row in zip(batch, cands)], dtype=np.int8)
            keep_users = (labels.sum(axis=1) > 0)  # users whose next items were not reachable teach the ranker nothing
            x = builder.build(batch, cands, feats).reshape(len(batch), cands.shape[1], -1)
            for b in np.flatnonzero(keep_users):
                valid = cands[b] > 0
                x_parts.append(x[b][valid])
                y_parts.append(labels[b][valid])
                u_parts.append(np.full(int(valid.sum()), batch[b]))
                i_parts.append(cands[b][valid])
        if len(x_parts) < 20:
            raise Unsupported("too few recent users with a reachable next item to train a re-ranker")
        self.fit_info = {"train_users": len(x_parts), "label_users": int(len(users)),
                         "train_recall_at_candidates": float(len(x_parts) / max(len(users), 1))}
        return np.concatenate(x_parts), np.concatenate(y_parts), np.concatenate(u_parts), np.concatenate(i_parts)

    def prepare_scoring(self) -> None:
        self.generators = Generators(self.cfg, self.ease_cfg, self.knn_cfg).fit(self.data)
        self.features = FeatureBuilder(self.data, self.text)
        self._cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def _candidates_and_features(self, users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = tuple(int(u) for u in users)
        if key not in self._cache:  # the validation monitor scores the same users after every epoch
            cands, feats = self.generators.candidates(users)
            self._cache = {key: (cands, self.features.build(users, cands, feats))}
        return self._cache[key]

    def candidate_sets(self, users: np.ndarray) -> list[np.ndarray]:
        cands, _ = self._candidates_and_features(users)
        return [row[row > 0] for row in cands]

    def score_users(self, users: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        cands, x = self._candidates_and_features(np.asarray(users))
        ranked = self.rank(np.repeat(users, cands.shape[1]), cands.ravel(), x).reshape(cands.shape)
        out = np.full((len(users), self.n_items + 1), NEG_INF, dtype=np.float32)
        rows = np.repeat(np.arange(len(users)), cands.shape[1]).reshape(cands.shape)
        valid = cands > 0
        out[rows[valid], cands[valid]] = ranked[valid]
        return out

    def explain(self, users, items, hist):
        out = []
        for user, row in zip(users, items):
            last = int(self.features.last_item[user])
            exps = []
            for item in row:
                if item <= 0:
                    exps.append(Explanation("none", "No recommendation."))
                elif last and self.features.covisit[last, int(item)] > 0:
                    exps.append(Explanation("personal", f"{self.spec.name}: people often go from {self.item_ids[last]} to this item.",
                                            [{"history_item": str(self.item_ids[last]), "covisits": float(self.features.covisit[last, int(item)])}]))
                else:
                    exps.append(Explanation("model", f"{self.spec.name}: ranked high among EASE, ItemKNN and trending candidates."))
            out.append(exps)
        return out

    def rank(self, users: np.ndarray, items: np.ndarray, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def _lightgbm_threads(cfg: dict[str, Any]) -> int:
    """On macOS, PyTorch and LightGBM ship different OpenMP runtimes; with both loaded, multi-threaded LightGBM
    crashes (segmentation fault). There, LightGBM runs on one thread; Linux boxes use all assigned threads."""
    if sys.platform == "darwin" and "torch" in sys.modules:
        return 1
    return int(cfg.get("threads") or 0) or -1


@register_method
class LGBMRerank(TwoStage):
    """Candidates from EASE/ItemKNN/popularity, re-ranked by a LightGBM LambdaRank model."""

    spec = MethodSpec(
        name="lgbm_rerank",
        tasks={Task.topn, Task.sequential},
        uses_history=True,
        upstream="LightGBM LambdaRank over in-repo candidates and features",
        cost_band="low",
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - part of the bench extra
            raise Unsupported("lightgbm is not installed") from exc
        self._setup(data, cfg)
        x, y, users, _ = self.training_table()
        order = np.argsort(users, kind="stable")
        x, y, users = x[order], y[order], users[order]
        unique, counts = np.unique(users, return_counts=True)
        rng = np.random.default_rng(int(cfg.get("seed", 42)))
        held = set(rng.permutation(unique)[: max(1, len(unique) // 10)].tolist())  # 10% of users for early stopping
        is_val = np.isin(users, list(held))
        group = lambda mask: np.unique(users[mask], return_counts=True)[1]  # noqa: E731 - rows are sorted by user
        self.model = lgb.LGBMRanker(
            objective="lambdarank", n_estimators=int(cfg.get("lgbm_trees", 500)), learning_rate=float(cfg.get("lgbm_lr", 0.05)),
            num_leaves=int(cfg.get("lgbm_leaves", 31)), min_child_samples=int(cfg.get("lgbm_min_child", 20)),
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=int(cfg.get("seed", 42)),
            n_jobs=_lightgbm_threads(cfg), verbose=-1,
        )
        self.model.fit(x[~is_val], y[~is_val], group=group(~is_val), eval_set=[(x[is_val], y[is_val])], eval_group=[group(is_val)],
                       eval_at=[10], callbacks=[lgb.early_stopping(50, verbose=False)])
        self.fit_info.update({"trees": int(self.model.best_iteration_ or self.model.n_estimators)})
        self.prepare_scoring()

    def rank(self, users: np.ndarray, items: np.ndarray, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x).astype(np.float32)


@register_method
class DCNV2Rerank(TwoStage):
    """The same candidates and features, re-ranked by DCN-V2 (cross network over ids, category and features)."""

    spec = MethodSpec(
        name="dcnv2_rerank",
        tasks={Task.topn, Task.sequential},
        uses_history=True,
        needs_torch=True,
        upstream="FuxiCTR 2.3 CrossNetV2 over in-repo candidates and features",
        cost_band="low",
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        import torch
        import torch.nn.functional as F
        from torch import nn

        from recbench.methods._torch import resolve_device, steps_per_epoch, train_epochs
        from recbench.methods.dcnv2 import category_index

        try:
            from fuxictr.pytorch.layers.interactions.cross_net import CrossNetV2
        except ImportError as exc:  # pragma: no cover
            raise Unsupported("fuxictr is not installed") from exc
        self._setup(data, cfg)
        self.device = resolve_device(cfg)
        x, y, users, items = self.training_table()
        self.mean, self.std = x.mean(axis=0), x.std(axis=0) + 1e-6
        dim = int(cfg.get("dim", 32))
        n_users, n_items, n_feats = self.n_users, self.n_items, x.shape[1]
        categories = torch.as_tensor(category_index(data.item_category)[:, 0])

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.user = nn.Embedding(n_users + 1, dim)
                self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)
                self.category = nn.Embedding(4096, dim, padding_idx=0)
                self.register_buffer("cat", categories)
                self.dense = nn.Linear(n_feats, dim)
                width = 4 * dim
                self.cross = CrossNetV2(width, max(int(cfg.get("layers", 2)), 1))
                self.deep = nn.Sequential(nn.Linear(width, 2 * dim), nn.ReLU(), nn.Dropout(float(cfg.get("dropout", 0.1))), nn.Linear(2 * dim, dim), nn.ReLU())
                self.head = nn.Linear(width + dim, 1)

            def forward(self, u, i, feats):
                x0 = torch.cat([self.user(u), self.item(i), self.category(self.cat[i]), self.dense(feats)], dim=-1)
                return self.head(torch.cat([self.cross(x0), self.deep(x0)], dim=-1)).squeeze(-1)

        self.net = Net().to(self.device)
        x_norm = torch.as_tensor((x - self.mean) / self.std, device=self.device)
        tensors = (torch.as_tensor(users, device=self.device), torch.as_tensor(items, device=self.device), x_norm,
                   torch.as_tensor(y, dtype=torch.float32, device=self.device))
        batch_size = int(cfg.get("batch_size", 1024))
        rng = np.random.default_rng(int(cfg.get("seed", 42)))

        def batches():  # random (user, candidate) rows, sampled with replacement
            while True:
                pick = torch.as_tensor(rng.integers(0, len(y), batch_size), device=self.device)
                yield tuple(t[pick] for t in tensors)

        def loss(batch):
            u, i, feats, labels = batch
            return F.binary_cross_entropy_with_logits(self.net(u, i, feats), labels)

        self.prepare_scoring()  # the validation monitor (on folds) scores through the full two-stage pipeline
        self.fit_info.update(train_epochs(self.net, batches(), steps_per_epoch(len(y), batch_size), loss, cfg, owner=self))

    def rank(self, users: np.ndarray, items: np.ndarray, x: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            feats = torch.as_tensor((x - self.mean) / self.std, device=self.device)
            return self.net(torch.as_tensor(users, device=self.device), torch.as_tensor(items, device=self.device), feats).float().cpu().numpy()
