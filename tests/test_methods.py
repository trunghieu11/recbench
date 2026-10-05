"""Methods: every one runs; sequence models read the right end of the history; reference parity checks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import FAST_CFG
from recbench.data import TrainView
from recbench.evaluation import EvalSplit, Evaluator
from recbench.registry import ensure_loaded

LOCAL_METHODS = sorted(n for n, cls in ensure_loaded().methods.items() if not cls.spec.managed and not cls.spec.requires_images)


@pytest.mark.parametrize("name", LOCAL_METHODS)
def test_every_method_fits_and_evaluates(toy, name):
    if name == "sansa" and sys.platform == "darwin":
        pytest.skip("SuiteSparse cannot share a process with PyTorch on macOS; test_sansa_ranks_like_exact_ease "
                    "fits SANSA in a child process instead")
    view, split = toy
    method = ensure_loaded().create_method(name)
    method.fit(view, dict(FAST_CFG))
    result = Evaluator(split, view, dict(FAST_CFG)).run(method)
    assert not result.errors, result.errors
    assert 0.0 <= result.metrics["ndcg_at_10"] <= 1.0
    assert np.isfinite(result.metrics["score_seconds_per_1k_users"])


GRU4REC_COPY = {"gru4rec_loss": "cross-entropy", "gru4rec_lr": 0.1, "gru4rec_dropout_embed": 0.0, "gru4rec_dropout_hidden": 0.0,
                "gru4rec_hidden": 64, "gru4rec_batch_size": 32, "gru4rec_n_sample": 32}


@pytest.mark.slow
@pytest.mark.parametrize("name,minimum,extra", [("sasrec", 0.9, {}), ("hstu", 0.9, {}), ("tiger_lite", 0.9, {}), ("bert4rec", 0.6, {}),
                                                ("gru4rec", 0.9, GRU4REC_COPY)])
def test_sequence_models_solve_the_copy_task(copy_split, name, minimum, extra):
    """Every user walks i -> i+1 -> ...; reading the wrong end of the history makes this unsolvable."""
    view, split = TrainView(copy_split), EvalSplit(copy_split)
    cfg = {**FAST_CFG, "dim": 32, "max_steps": 300, "max_epochs": 80, "lr": 5e-3, "batch_size": 64, "dropout": 0.0, "seed": 0, **extra}
    method = ensure_loaded().create_method(name)
    method.fit(view, cfg)
    users = split.users_of(warm=True)
    scores = method.full_scores(users, view.history_batch(users, cfg["seq_len"]))
    scores[:, 0] = -np.inf
    accuracy = np.mean(scores.argmax(axis=1) == split.next_items(users, "allow_repeats"))
    assert accuracy >= minimum, f"{name} next-item accuracy {accuracy:.2f}"


def test_hstu_attention_matches_metas_reference_op():
    repo = Path(__file__).resolve().parents[1] / "third_party" / "generative-recommenders"
    if not (repo / "generative_recommenders").is_dir():
        pytest.skip("third_party/generative-recommenders not fetched")
    sys.path.insert(0, str(repo))

    def jagged_to_padded_dense(values, offsets, max_lengths, padding_value=0.0):
        offsets, n = offsets[0], int(max_lengths[0])
        out = values.new_full((offsets.numel() - 1, n, values.shape[-1]), padding_value)
        for b in range(offsets.numel() - 1):
            seg = values[offsets[b] : offsets[b + 1]]
            out[b, : len(seg)] = seg
        return out

    def dense_to_jagged(dense, offsets, total=None):
        offsets = offsets[0]
        return (torch.cat([dense[b, : int(offsets[b + 1] - offsets[b])] for b in range(offsets.numel() - 1)]),)

    torch.ops.fbgemm.jagged_to_padded_dense = jagged_to_padded_dense
    torch.ops.fbgemm.dense_to_jagged = dense_to_jagged
    from generative_recommenders.ops.pytorch.pt_hstu_attention import pytorch_hstu_mha

    from recbench.methods.hstu import hstu_attention

    torch.manual_seed(0)
    batch, heads, length, dim = 2, 2, 6, 4
    q, k, v = (torch.randn(batch, heads, length, dim) for _ in range(3))
    ours = hstu_attention(q, k, v, torch.ones(batch, length, dtype=torch.bool))
    flat = lambda t: t.transpose(1, 2).reshape(batch * length, heads, dim)  # noqa: E731
    offsets = torch.arange(0, (batch + 1) * length, length)
    theirs = pytorch_hstu_mha(length, 1.0, flat(q), flat(k), flat(v), offsets, causal=True, training=False)
    assert torch.allclose(flat(ours), theirs, atol=1e-5)


def test_bert4rec_inputs_match_recboles_own_training_rows(toy):
    view, _ = toy
    method = ensure_loaded().create_method("bert4rec")
    method.fit(view, {**FAST_CFG, "max_steps": 2})
    dataset = method.rb_data
    seq_field, len_field = method.model.ITEM_SEQ, method.model.ITEM_SEQ_LEN
    user = int(view.warm_users()[0])
    history = view.user_items(user)
    from recbench.data import HistoryBatch

    prefix = history[:-1][-method.seq_len :]
    padded = np.zeros((1, method.seq_len), dtype=np.int64)
    padded[0, -len(prefix) :] = prefix
    ours, lengths = method._sequences(HistoryBatch(padded, np.array([len(prefix)])))
    rb_user = method.rb_user[user]
    rows = (dataset.inter_feat[dataset.uid_field] == rb_user).nonzero().ravel()
    last = rows[-1]  # RecBole keeps a user's augmented rows in time order; the last one predicts the final item
    assert torch.equal(ours[0].cpu(), dataset.inter_feat[seq_field][last].cpu())
    assert int(lengths[0]) == int(dataset.inter_feat[len_field][last])


def test_ease_matches_the_closed_form(toy):
    view, _ = toy
    method = ensure_loaded().create_method("ease")
    method.fit(view, {"ease_lambda": 10.0})
    x = np.asarray(view.seen[:, method.kept].todense(), dtype=np.float64)
    p = np.linalg.inv(x.T @ x + 10.0 * np.eye(x.shape[1]))
    b = -p / np.diag(p)
    np.fill_diagonal(b, 0.0)
    assert np.allclose(method.weights, b, atol=1e-4)


def test_itemknn_matches_implicit_cosine(toy):
    view, _ = toy
    implicit = pytest.importorskip("implicit")
    ours = ensure_loaded().create_method("itemknn")
    ours.fit(view, {"knn_neighbors": view.n_items + 1, "knn_shrink": 0.0})
    reference = implicit.nearest_neighbours.CosineRecommender(K=view.n_items + 1)
    reference.fit(view.seen.tocsr().astype(np.float64), show_progress=False)
    users = view.warm_users()[:10]
    mine = ours.score_users(users, None)
    theirs = np.asarray((view.seen[users] @ reference.similarity.T).todense())
    np.fill_diagonal(theirs[:, : 0], 0)
    seen_mask = np.asarray(view.seen[users].todense()) > 0
    # Compare the ranking of unseen items (implicit keeps self-similarity, which only affects seen items).
    for row in range(len(users)):
        candidates = np.flatnonzero(~seen_mask[row])[1:]
        assert np.corrcoef(mine[row, candidates], theirs[row, candidates])[0, 1] > 0.99
