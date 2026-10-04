"""Linear item-to-item models beyond EASE: PureSVD, SLIM-ElasticNet, SANSA.

All three score a user as (their interaction vector) x (an item x item matrix):
- PureSVD (Cremonesi et al. 2010): the matrix is V V^T from a truncated SVD (a low-rank projection).
- SLIM (Ning & Karypis 2011): a sparse, non-negative matrix learned by one ElasticNet regression per item.
- SANSA (Spisak et al. 2023): a sparse approximation of EASE's closed form, so the catalog needs no cap.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import scipy.sparse as sp

from recbench.data import HistoryBatch, TrainView
from recbench.methods._explain import contribution_explanations
from recbench.methods.baselines import BASELINE_TASKS, item_cosine_topk
from recbench.protocol import NEG_INF, MethodSpec, Recommender, Unsupported
from recbench.registry import register_method


@register_method
class PureSVD(Recommender):
    """PureSVD: keep the top `factors` singular vectors V of the user x item matrix; score(u) = x_u V V^T."""

    spec = MethodSpec(
        name="puresvd",
        tasks=BASELINE_TASKS,
        uses_history=True,
        upstream="scikit-learn randomized_svd (in-repo wrapper)",
        cost_band="low",
        deterministic=True,
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        from sklearn.utils.extmath import randomized_svd

        self.bind(data)
        self.seen = data.weighted_matrix(cfg.get("decay_half_life_days")).astype(np.float32)
        factors = max(1, min(int(cfg.get("svd_factors", 128)), min(self.seen.shape) - 1))
        _, _, vt = randomized_svd(self.seen, n_components=factors, n_iter=5, random_state=int(cfg.get("seed", 42)))
        self.v = np.ascontiguousarray(vt.T, dtype=np.float32)  # items x factors

    def score_users(self, users: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        out = np.asarray(self.seen[users] @ self.v, dtype=np.float32) @ self.v.T
        out[:, 0] = NEG_INF
        return out

    def item_embeddings(self) -> np.ndarray:
        return self.v


def _slim_column(x: sp.csc_matrix, j: int, features: np.ndarray, alpha: float, l1_ratio: float, max_iter: int) -> tuple[np.ndarray, np.ndarray]:
    """One SLIM regression: predict item j's column from its candidate neighbours (non-negative weights)."""
    from sklearn.linear_model import ElasticNet

    features = features[features != j]
    if len(features) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    target = x[:, j].toarray().ravel()
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, positive=True, fit_intercept=False, copy_X=False,
                       precompute=True, selection="random", max_iter=max_iter, tol=1e-4, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence warnings at small max_iter are expected
        model.fit(x[:, features], target)
    keep = model.coef_ > 0
    return features[keep].astype(np.int64), model.coef_[keep].astype(np.float32)


@register_method
class SLIM(Recommender):
    """SLIM-ElasticNet with neighbour pre-selection (fsSLIM): each item's weights come from an ElasticNet
    regression on its `slim_neighbors` most similar items only, which makes it fast on large catalogs."""

    spec = MethodSpec(
        name="slim",
        tasks=BASELINE_TASKS,
        uses_history=True,
        upstream="scikit-learn ElasticNet per item, parallel with joblib (in-repo)",
        cost_band="low",
        deterministic=True,
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        from joblib import Parallel, delayed

        self.bind(data)
        self.seen = data.weighted_matrix(cfg.get("decay_half_life_days")).astype(np.float32)
        neighbours = item_cosine_topk(self.seen, int(cfg.get("slim_neighbors", 100)), 0.0)
        x = self.seen.tocsc()
        warm = np.flatnonzero(data.item_pop > 0)
        warm = warm[warm > 0]
        alpha, l1_ratio = float(cfg.get("slim_alpha", 1e-3)), float(cfg.get("slim_l1_ratio", 0.1))
        jobs = int(cfg.get("threads") or -1)
        results = Parallel(n_jobs=jobs, batch_size=256, prefer="threads")(
            delayed(_slim_column)(x, int(j), neighbours[j].indices, alpha, l1_ratio, int(cfg.get("slim_max_iter", 100))) for j in warm
        )
        rows = np.concatenate([idx for idx, _ in results]) if results else np.zeros(0, dtype=np.int64)
        vals = np.concatenate([val for _, val in results]) if results else np.zeros(0, dtype=np.float32)
        cols = np.concatenate([np.full(len(idx), j, dtype=np.int64) for j, (idx, _) in zip(warm, results)]) if results else rows
        n = self.n_items + 1
        self.weights = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))  # W[i, j]: how much history item i predicts j
        self.fit_info = {"nonzeros": int(self.weights.nnz)}

    def score_users(self, users: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        out = np.asarray((self.seen[users] @ self.weights).todense(), dtype=np.float32)
        out[:, 0] = NEG_INF
        return out

    def explain(self, users, items, hist):
        return contribution_explanations(users, items, lambda u: self.seen[u].indices,
                                         lambda history, j: self.weights[history, j].toarray(), self.item_ids, "SLIM")


@register_method
class SANSA(Recommender):
    """SANSA: EASE's closed form computed approximately with a sparse LDL^T factorisation, so it scales to
    large catalogs (no item cap). Uses the `sansa` package (needs SuiteSparse)."""

    spec = MethodSpec(
        name="sansa",
        tasks=BASELINE_TASKS,
        uses_history=True,
        upstream="sansa (Spisak et al. 2023)",
        cost_band="low",
        deterministic=True,
    )

    def fit(self, data: TrainView, cfg: dict[str, Any]) -> None:
        try:
            from sansa import CHOLMODGramianFactorizerConfig, ICFGramianFactorizerConfig, SANSAConfig, UMRUnitLowerTriangleInverterConfig
            from sansa import SANSA as SansaModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise Unsupported("the `sansa` package is not installed (pip install 'recbench[sansa]'; needs SuiteSparse)") from exc
        logging.getLogger("sansa").setLevel(logging.WARNING)
        # macOS: PyTorch's OpenMP runtime and the one Homebrew's SuiteSparse loads abort the process together
        # once CHOLMOD runs in parallel. Queue and benchmark runs import only this module, so PyTorch is not
        # loaded (see recbench.methods); in a notebook, do not import PyTorch before fitting SANSA.
        self.bind(data)
        self.seen = data.weighted_matrix(cfg.get("decay_half_life_days")).astype(np.float32).tocsr()
        factorizer = (ICFGramianFactorizerConfig() if cfg.get("sansa_factorizer", "cholmod") == "icf"
                      else CHOLMODGramianFactorizerConfig())
        # `sansa_weights_per_item` (when set) fixes the number of weights per item, so one search range suits
        # catalogs of any size; `sansa_density` is the package's own setting, a share of all item pairs.
        per_item = cfg.get("sansa_weights_per_item")
        density = min(1.0, float(per_item) / max(self.n_items, 1)) if per_item else float(cfg.get("sansa_density", 1e-3))
        model = SansaModel(SANSAConfig(
            l2=float(cfg.get("sansa_lambda", 500.0)),
            weight_matrix_density=density,
            gramian_factorizer_config=factorizer,
            lower_triangle_inverter_config=UMRUnitLowerTriangleInverterConfig(),  # the package defaults
        ))
        model.fit(self.seen)
        self.w1, self.w2 = model.weights
        self.fit_info = {"nonzeros": int(self.w1.nnz + self.w2.nnz), "density": density}

    def score_users(self, users: np.ndarray, hist: HistoryBatch) -> np.ndarray:
        out = np.asarray(((self.seen[users] @ self.w1) @ self.w2).todense(), dtype=np.float32)
        out[:, 0] = NEG_INF
        return out
