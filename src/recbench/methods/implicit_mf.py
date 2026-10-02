"""Matrix factorisation for implicit feedback, via the `implicit` library (Ben Frederickson).

- iALS (Hu, Koren & Volinsky 2008): weighted least squares on "did interact" with
  confidence 1 + alpha * count, solved by alternating closed-form updates.
- BPR-MF (Rendle et al. 2009): pairwise ranking loss, learned with SGD over
  (user, interacted item, random item) triples.
Both learn user vectors p_u and item vectors q_i; score(u, i) = p_u . q_i.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from recbench.data import HistoryBatch, TrainView
from recbench.methods._explain import embedding_explanations
from recbench.protocol import NEG_INF, MethodSpec, Recommender, Task, Unsupported
from recbench.registry import register_method

MF_TASKS = {Task.topn, Task.sequential, Task.similar_items}


class _ImplicitMF(Recommender):
    def _factors(self, model) -> None:
        # CPU models expose numpy factors. BPR appends a bias column to both matrices,
        # so user . item already includes the item bias.
        self.user_factors = np.asarray(model.user_factors, dtype=np.float32)
        self.item_factors = np.asarray(model.item_factors, dtype=np.float32)

    def score_users(self, users: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        scores = self.user_factors[users] @ self.item_factors.T
        scores[:, 0] = NEG_INF
        return scores

    def item_embeddings(self) -> np.ndarray:
        return self.item_factors

    def explain(self, users, items, hist):
        return embedding_explanations(self.item_factors, users, items, hist, self.item_ids, self.spec.name)


def _implicit():
    try:
        import implicit
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise Unsupported("the `implicit` package is not installed (pip install implicit==0.7.3)") from exc
    return implicit


@register_method
class IALS(_ImplicitMF):
    spec = MethodSpec(name="ials", tasks=MF_TASKS, upstream="implicit 0.7.3 AlternatingLeastSquares", cost_band="low")

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self.bind(data)
        implicit = _implicit()
        model = implicit.als.AlternatingLeastSquares(
            factors=int(cfg.get("dim", 64)),
            regularization=float(cfg.get("ials_reg", 0.01)),
            alpha=float(cfg.get("ials_alpha", 10.0)),
            iterations=int(cfg.get("ials_iterations", 15)),
            random_state=int(cfg.get("seed", 42)),
            use_gpu=False,
        )
        model.fit(data.interaction_counts.tocsr(), show_progress=False)
        self._factors(model)


@register_method
class BPRMF(_ImplicitMF):
    spec = MethodSpec(name="bpr_mf", tasks=MF_TASKS, upstream="implicit 0.7.3 BayesianPersonalizedRanking", cost_band="low")

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        self.bind(data)
        implicit = _implicit()
        model = implicit.bpr.BayesianPersonalizedRanking(
            factors=int(cfg.get("dim", 64)),
            learning_rate=float(cfg.get("bpr_lr", 0.01)),
            regularization=float(cfg.get("bpr_reg", 0.01)),
            iterations=int(cfg.get("bpr_iterations", 100)),
            random_state=int(cfg.get("seed", 42)),
            use_gpu=False,
        )
        model.fit(data.seen.tocsr(), show_progress=False)
        self._factors(model)
