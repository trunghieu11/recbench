"""One quick-tier job: tune a method on a dataset's validation fold, then test the best settings once.

A job (one method on one dataset):
1. runs up to `trials` configurations on the validation fold '<tier>-val', each in its own child process.
   The sampler is seeded TPE, and every method gets the same budget. Each configuration is scored on the
   same fixed sample of `search_users` validation users. Iterative models stop early on that fold.
2. stops searching when the time left would not fit one more trial plus the final run (the job cap
   includes everything);
3. runs the best configuration once on '<tier>' (the real test split), reusing the best epoch count, so
   the test split is touched exactly once;
4. writes a summary to runs/tuning/<tier>/<dataset>/<method>.json.

Finished trials live in an Optuna journal and finished runs in MLflow, so an interrupted job resumes
where it stopped.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recbench.config import config_hash, method_config
from recbench.registry import ensure_loaded
from recbench.runner import EXPERIMENT, _mlflow, data_root, repo_root, run_pair
from recbench.tuning.spaces import MethodSpace

METRIC = "ndcg_at_10"


@dataclass
class JobSettings:
    tier: str = "quick"
    trials: int = 10
    search_users: int = 3000
    cap_minutes: float = 180.0
    final_factor: float = 1.5  # until measured, the final run is assumed to take this many typical trials
    seed: int = 42
    space_version: int = 1


def tuning_dir() -> Path:
    path = repo_root() / "runs" / "tuning"
    path.mkdir(parents=True, exist_ok=True)
    return path


def summary_path(tier: str, dataset: str, method: str) -> Path:
    return tuning_dir() / tier / dataset / f"{method}.json"


def read_summary(tier: str, dataset: str, method: str) -> dict[str, Any] | None:
    path = summary_path(tier, dataset, method)
    return json.loads(path.read_text()) if path.exists() else None


def _write_summary(summary: dict[str, Any]) -> Path:
    path = summary_path(summary["tier"], summary["dataset"], summary["method"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, default=str))
    tmp.replace(path)
    return path


def _storage():
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    return JournalStorage(JournalFileBackend(str(tuning_dir() / "journal.log")))


def _finished_metric(dataset: str, method: str, cfg: dict[str, Any], split_dir: Path) -> float | None:
    """The metric of an identical run that already finished (a resumed job, or a repeated configuration)."""
    from recbench.data import TrainView

    spec = ensure_loaded().methods[method].spec
    run_hash = config_hash(method_config(cfg, method), method, TrainView(split_dir).split_hash, spec.impl_version)
    mlflow = _mlflow()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    frame = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.config_hash = '{run_hash}' and tags.status = 'finished'",
        max_results=1,
    )
    column = f"metrics.{METRIC}"
    return None if frame.empty or column not in frame else float(frame[column].iloc[0])


def _outcome_value(outcome: dict[str, Any], dataset: str, method: str, cfg: dict[str, Any], split_dir: Path) -> float | None:
    if outcome.get("status") == "finished":
        return (outcome.get("metrics") or {}).get(METRIC)
    if outcome.get("status") == "skipped_existing":
        return _finished_metric(dataset, method, cfg, split_dir)
    return None


def _base_cfg(resolved: dict[str, Any], method: str, tuned_keys: set[str]) -> dict[str, Any]:
    """The resolved config without per-method overrides of tuned keys (they would silently win otherwise)."""
    cfg = dict(resolved)
    own = dict((resolved.get("method_params") or {}).get(method) or {})
    cfg["method_params"] = {**(resolved.get("method_params") or {}), method: {k: v for k, v in own.items() if k not in tuned_keys}}
    return cfg


def run_job(
    dataset: str,
    method: str,
    resolved: dict[str, Any],
    settings: JobSettings,
    space: MethodSpace | None,
    *,
    child_env: dict[str, str] | None = None,
    isolate: bool = True,
) -> dict[str, Any]:
    """Tune `method` on `dataset` and evaluate the best configuration once. Returns the job summary."""
    import optuna
    from optuna.trial import TrialState

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    began = time.time()
    deadline = began + settings.cap_minutes * 60
    root = data_root()
    val_dir = root / "splits" / dataset / f"{settings.tier}-val"
    test_dir = root / "splits" / dataset / settings.tier
    summary: dict[str, Any] = {
        "tier": settings.tier, "dataset": dataset, "method": method, "started": began,
        "cap_minutes": settings.cap_minutes, "trials_budget": 0, "trials": [],
    }
    missing = [str(p) for p in (val_dir, test_dir) if not (p / "meta.json").exists()]
    if missing:
        summary.update(status="missing_split", reason=f"prepare these splits first: {missing}", ended=time.time())
        _write_summary(summary)
        return summary

    space = space or MethodSpace(method)
    n_trials = space.trials if space.trials is not None else (settings.trials if space.params else 1)
    summary["trials_budget"] = n_trials
    base = _base_cfg(resolved, method, set(space.params) | set(space.fixed))
    base.update({"tuning": "tuned", "export_bundles": False, "resume": True})
    if child_env:
        base["child_env"] = dict(child_env)
    study = optuna.create_study(
        study_name=f"{settings.tier}/{dataset}/{method}/v{settings.space_version}/s{settings.seed}",
        storage=_storage(),
        sampler=optuna.samplers.TPESampler(seed=settings.seed, n_startup_trials=min(5, n_trials)),
        direction="maximize",
        load_if_exists=True,
    )
    stop_reason = "budget"
    while True:
        done = [t for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.FAIL)]
        if len(done) >= n_trials:
            break
        durations = [t.user_attrs["seconds"] for t in done if "seconds" in t.user_attrs]
        typical = statistics.median(durations) if durations else None
        remaining = deadline - time.time()
        reserve = typical * settings.final_factor if typical else 0.25 * settings.cap_minutes * 60
        if (typical and remaining < typical + reserve) or remaining - reserve < 60:
            stop_reason = "time"
            break
        trial = study.ask()
        params = space.sample(trial)
        trial_budget = remaining - reserve
        cfg = {**base, **params, "stage": "search", "trial": trial.number, "max_eval_users": settings.search_users,
               "timeout_minutes": trial_budget / 60, "fit_deadline": time.time() + 0.8 * trial_budget}
        t0 = time.time()
        outcome = run_pair(val_dir, method, cfg, isolate=isolate)
        seconds = time.time() - t0
        value = _outcome_value(outcome, dataset, method, cfg, val_dir)
        trial.set_user_attr("seconds", seconds)
        trial.set_user_attr("status", outcome.get("status"))
        fit = outcome.get("fit") or {}
        if fit.get("best_epoch"):
            trial.set_user_attr("best_epoch", int(fit["best_epoch"]))
        if value is None:
            trial.set_user_attr("reason", str(outcome.get("reason", ""))[:300])
            study.tell(trial, state=TrialState.FAIL)
        else:
            study.tell(trial, float(value))

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    summary["trials"] = [
        {"number": t.number, "state": t.state.name, "value": t.value, "params": t.params, **t.user_attrs} for t in study.trials
        if t.state in (TrialState.COMPLETE, TrialState.FAIL)
    ]
    summary["search_stopped"] = stop_reason
    if not completed:
        failed = [t for t in study.trials if t.state == TrialState.FAIL]
        timeouts = [t for t in failed if t.user_attrs.get("status") == "timeout"]
        summary.update(status="over_budget" if (stop_reason == "time" or timeouts) else "failed",
                       reason=(failed[-1].user_attrs.get("reason") if failed else "no trial fitted in the time cap"), ended=time.time())
        _write_summary(summary)
        return summary

    best = study.best_trial
    best_params = {**space.fixed, **best.params}
    summary.update(best_trial=best.number, best_params=best_params, best_val=best.value, best_epoch=best.user_attrs.get("best_epoch"))
    remaining = deadline - time.time()
    final_cfg = {**base, **best_params, "stage": "final", "timeout_minutes": max(remaining, 60) / 60}
    if best.user_attrs.get("best_epoch"):
        final_cfg["epochs"] = int(best.user_attrs["best_epoch"])
    t0 = time.time()
    final = run_pair(test_dir, method, final_cfg, isolate=isolate)
    summary["final_seconds"] = time.time() - t0
    summary["final_status"] = final.get("status")
    test_value = _outcome_value(final, dataset, method, final_cfg, test_dir)
    summary["test"] = {**(final.get("metrics") or {}), METRIC: test_value} if test_value is not None else final.get("metrics")
    if test_value is not None:
        summary["status"] = "finished"
    else:
        summary["status"] = "over_budget" if final.get("status") == "timeout" else "failed"
        summary["reason"] = final.get("reason", final.get("status"))
    summary["ended"] = time.time()
    _write_summary(summary)
    return summary
