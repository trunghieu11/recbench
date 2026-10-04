"""Checks for method variants. The template tests (labs/templates/test_variant_template.py) and
tests/test_lab_defaults.py use them; they run on the small toy data of tests/conftest.py in a second or two.

A variant is a new setting of an existing method (for example `ease_variant: edlae`) whose default keeps today's
behaviour. These checks answer four questions about it: does it keep the scores contract, is the default unchanged,
does the variant change anything at all, and does it still learn?
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Small, fast settings for the toy data; everything else stays at the method's default.
TOY_SETTINGS: dict[str, Any] = {"seed": 42, "ease_backend": "numpy", "threads": 1, "dim": 16, "ials_iterations": 5, "bpr_iterations": 20,
                                "rerank_candidates": 50, "lgbm_trees": 50}
N_USERS = 30


def fit(method: str, view, **settings: Any):
    """A fitted model of `method` on `view` with TOY_SETTINGS plus `settings`."""
    from recbench.registry import ensure_loaded

    model = ensure_loaded().create_method(method)
    model.fit(view, {**TOY_SETTINGS, **settings})
    return model


def users_of(view, n: int = N_USERS) -> np.ndarray:
    return np.asarray(view.warm_users()[:n], dtype=np.int64)


def scores(model, view, users: np.ndarray | None = None) -> np.ndarray:
    users = users_of(view) if users is None else users
    return np.asarray(model.score_users(users, view.history_batch(users, 10)))


def check_scores(model, view, users: np.ndarray | None = None) -> np.ndarray:
    """The contract every method keeps: one row per user, one column per item plus the padding column 0, floats, no NaN, and
    at least one finite score per user. Returns the scores."""
    users = users_of(view) if users is None else users
    s = scores(model, view, users)
    assert s.shape == (len(users), view.n_items + 1), f"scores have shape {s.shape}, expected ({len(users)}, {view.n_items + 1})"
    assert np.issubdtype(s.dtype, np.floating), f"scores must be floats, got {s.dtype}"
    assert not np.isnan(s).any(), "scores contain NaN (look for 0/0, log(0) or a negative number to a fractional power)"
    assert np.isfinite(s[:, 1:]).any(axis=1).all(), "some users get no finite score at all"
    return s


def check_same_scores(method: str, view, settings_a: dict[str, Any], settings_b: dict[str, Any]) -> None:
    """Two settings that must mean the same model, for example the variant's default spelled out."""
    a, b = scores(fit(method, view, **settings_a), view), scores(fit(method, view, **settings_b), view)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6, err_msg=f"{settings_b} should give the same model as {settings_a}")


def check_differs(method: str, view, settings_a: dict[str, Any], settings_b: dict[str, Any]) -> None:
    """Two settings that must give different models: identical scores usually mean the new setting is never read."""
    a, b = scores(fit(method, view, **settings_a), view), scores(fit(method, view, **settings_b), view)
    assert not np.allclose(a, b, rtol=1e-6, atol=1e-9), f"{settings_b} gives exactly the scores of {settings_a}: is the setting read?"


def check_deterministic(method: str, view, **settings: Any) -> None:
    """Fitting twice with the same settings gives the same scores."""
    a, b = scores(fit(method, view, **settings), view), scores(fit(method, view, **settings), view)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6, err_msg="two fits with the same settings differ")


def toy_ndcg(method: str, view, split, **settings: Any) -> float:
    """NDCG@10 on the toy split's evaluation users (to check that a variant still learns something)."""
    from recbench.evaluation import Evaluator

    model = fit(method, view, **settings)
    return float(Evaluator(split, view, {**TOY_SETTINGS, **settings}).run(model).metrics["ndcg_at_10"])


# ----- golden defaults (tests/test_lab_defaults.py) ---------------------------------------------------------------------

def golden_record(method: str, view) -> dict[str, Any]:
    """What tests/golden/lab_defaults.json stores for a method at default settings: for the first users, the 10 highest
    scores and their items, plus the method's impl_version."""
    from recbench.registry import ensure_loaded

    model = fit(method, view)
    users = users_of(view)
    s = scores(model, view, users).astype(np.float64)
    s[:, 0] = -np.inf
    top = np.argsort(-s, axis=1, kind="stable")[:, :10]
    return {"impl_version": ensure_loaded().methods[method].spec.impl_version, "users": users.tolist(), "items": top.tolist(),
            "scores": np.take_along_axis(s, top, axis=1).round(7).tolist()}


def compare_golden(now: dict[str, Any], golden: dict[str, Any], rtol: float = 1e-4, atol: float = 1e-6) -> list[str]:
    """Differences between a fresh golden record and the stored one ([] when the behaviour is unchanged). The 10 best
    scores must match; so must the items, except among scores that tie."""
    problems = []
    if now["users"] != golden["users"]:
        return ["the toy data changed (other users): regenerate the golden file"]
    for user, items_now, items_old, s_now, s_old in zip(now["users"], now["items"], golden["items"], now["scores"], golden["scores"]):
        s_now, s_old = np.asarray(s_now, dtype=float), np.asarray(s_old, dtype=float)
        if not np.allclose(s_now, s_old, rtol=rtol, atol=atol):
            problems.append(f"user {user}: top scores {np.round(s_now[:3], 5).tolist()}... were {np.round(s_old[:3], 5).tolist()}...")
            continue
        for pos, (a, b) in enumerate(zip(items_now, items_old)):
            tied = any(np.isclose(s_old[pos], s_old[q], rtol=rtol, atol=atol) for q in range(len(s_old)) if q != pos)
            if a != b and not tied:
                problems.append(f"user {user}: item at rank {pos + 1} is {a}, was {b}")
                break
    return problems
