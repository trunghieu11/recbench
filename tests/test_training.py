"""Training loops: epochs with early stopping on a validation fold only, losses, learning curves, user ids."""

from __future__ import annotations

import mlflow
import numpy as np
import pytest
import torch

from conftest import FAST_CFG
from recbench.data import TrainView
from recbench.evaluation import EvalSplit, Evaluator, ValidationMonitor
from recbench.methods.sasrec import SASRec
from recbench.methods.seq_trainer import next_item_loss
from recbench.runner import run_pair


def test_monitor_refuses_a_real_test_split(toy_split):
    with pytest.raises(ValueError, match="validation fold"):
        ValidationMonitor(EvalSplit(toy_split), TrainView(toy_split), dict(FAST_CFG))


def test_sasrec_stops_early_on_the_fold_and_keeps_its_best_epoch(toy_fold):
    data, split = TrainView(toy_fold), EvalSplit(toy_fold)
    cfg = {**FAST_CFG, "max_epochs": 6, "patience": 2}
    method = SASRec()
    method.monitor = ValidationMonitor(split, data, cfg)
    method.fit(data, cfg)
    info, curve = method.fit_info, method.fit_curve
    assert 1 <= info["epochs_run"] <= 6 and len(curve) == info["epochs_run"]
    assert all("val_ndcg_at_10" in point for point in curve)
    assert info["best_val_ndcg_at_10"] == max(point["val_ndcg_at_10"] for point in curve)
    assert curve[info["best_epoch"] - 1]["val_ndcg_at_10"] == info["best_val_ndcg_at_10"]


def test_fixed_epochs_never_look_at_validation(toy_fold):
    data = TrainView(toy_fold)
    method = SASRec()
    method.monitor = lambda *_: pytest.fail("the monitor must not run when the epoch count is fixed")
    method.fit(data, {**FAST_CFG, "epochs": 2})
    assert method.fit_info["epochs_run"] == 2


@pytest.mark.parametrize("mode", ["auto", "sampled", "bce"])
def test_next_item_loss_modes(mode):
    torch.manual_seed(0)
    hidden, weight = torch.randn(4, 5, 8, requires_grad=True), torch.randn(51, 8, requires_grad=True)
    targets = torch.randint(0, 51, (4, 5))
    loss = next_item_loss(hidden, weight, targets, mode=mode, n_negatives=16, logits_budget=1e9 if mode == "auto" else 0)
    loss.backward()
    assert torch.isfinite(loss) and hidden.grad is not None


def test_per_user_arrays_carry_user_ids(toy):
    data, split = toy
    from recbench.methods.baselines import MostPopular

    method = MostPopular()
    method.fit(data, dict(FAST_CFG))
    result = Evaluator(split, data, dict(FAST_CFG)).run(method)
    assert np.array_equal(result.per_user["user_idx"], np.sort(split.users_of(warm=True)))
    assert len(result.per_user["ndcg_at_10"]) == len(result.per_user["user_idx"])


def test_runner_logs_learning_curves_on_a_fold(toy_fold, tmp_path, monkeypatch):
    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    resolved = {**FAST_CFG, "tier": "full-val", "preset": "cpu", "hardware_name": "test", "resume": False, "continue_on_error": False,
                "timeout_minutes": 5, "managed_services": False, "export_bundles": False, "datasets": [], "methods": [],
                "method_params": {}, "max_epochs": 3, "stage": "search", "tuning": "tuned"}
    outcome = run_pair(toy_fold, "sasrec", resolved, isolate=False)
    assert outcome["status"] == "finished" and outcome["fit"]["epochs_run"] >= 1
    mlflow.set_tracking_uri(uri)
    run = mlflow.search_runs(experiment_names=["recbench"]).iloc[0]
    assert run["tags.stage"] == "search" and run["tags.tier"] == "full-val"
    history = mlflow.MlflowClient(uri).get_metric_history(run["run_id"], "curve/val_ndcg_at_10")
    assert len(history) == outcome["fit"]["epochs_run"]


def test_a_blown_up_final_run_restores_its_best_weights():
    """With a fixed epoch count nothing watches validation. A loss that jumps (training diverged) restores the
    lowest-loss epoch's weights and stops, as SASRec's final run on RetailRocket needed in the first bake-off."""
    from recbench.methods._torch import early_stopping_loop

    model = torch.nn.Linear(1, 1)

    def runner(losses):
        def run_epoch(epoch):
            with torch.no_grad():
                model.weight.fill_(float(epoch))  # the weights remember the epoch that wrote them
            return losses[epoch - 1]
        return run_epoch

    info = early_stopping_loop(model, runner([5.0, 4.0, 3.0, 9.0, 10.0]), {"epochs": 5})
    assert (info["stopped"], info["epochs_run"], info["best_epoch"]) == ("diverged", 4, 3)
    assert model.weight.item() == 3.0
    info = early_stopping_loop(model, runner([2.0, float("nan"), 1.0]), {"epochs": 3})
    assert (info["stopped"], info["best_epoch"], model.weight.item()) == ("diverged", 1, 1.0)
    # Negative losses (DirectAU's) and small wobbles are normal training, not a blow-up.
    info = early_stopping_loop(model, runner([-1.0, -2.0, -1.9, -2.5]), {"epochs": 4})
    assert (info["stopped"], info["epochs_run"], model.weight.item()) == ("max_epochs", 4, 4.0)
