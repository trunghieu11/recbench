"""One quick-tier job: tune a method on a dataset's validation fold, then test the best settings once.

A job (one method on one dataset):
1. runs up to `trials` configurations on the validation fold '<tier>-val', each in its own child process.
   The sampler is seeded TPE, and every method gets the same budget. Each configuration is scored on the
   same fixed sample of `search_users` validation users. Iterative models stop early on that fold.
2. stops searching when the time left would not fit one more trial plus the final run (the job cap
   includes everything);
3. runs the best configuration once on '<tier>' (the real test split), reusing the best epoch count, so
   the test split is touched exactly once. With `final_seeds` (the lab), methods whose training is random run that
   one configuration once per seed, and the test result is their average;
4. writes a summary to runs/tuning/<tier>/<dataset>/<method>.json.

A lab experiment is a job with a `label`: its summary is <method>@<label>.json and it has its own Optuna study. In
the lab workspace all of this lives under runs/lab/ (see recbench.paths).

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

from recbench.paths import runs_dir
from recbench.registry import ensure_loaded
from recbench.runner import EXPERIMENT, _mlflow, data_root, method_version, run_hash_for, run_pair
from recbench.tuning.spaces import MethodSpace

METRIC = "ndcg_at_10"
BASELINE = "baseline"  # the label of a method's plain job (no label in its file name)


@dataclass
class JobSettings:
    tier: str = "quick"
    trials: int = 10
    search_users: int = 3000
    cap_minutes: float = 180.0
    final_factor: float = 1.5  # until measured, the final run is assumed to take this many typical trials
    seed: int = 42
    space_version: int = 1
    force: dict[str, Any] | None = None  # settings that win over the search spaces (dry runs: {max_epochs: 3})
    final_seeds: list[int] | None = None  # the lab: test random methods once per seed, then average


def tuning_dir(workspace: str | None = None) -> Path:
    """runs/tuning, or runs/<workspace>/tuning (`workspace` reads another one; "" is the bake-off). Reading creates
    nothing; writers create the folders they need."""
    return runs_dir(workspace) / "tuning"


def summary_path(tier: str, dataset: str, method: str, stage: str = "tune", *, label: str | None = None,
                 workspace: str | None = None) -> Path:
    """runs/tuning/<tier>/<dataset>/<method>.json for a tuning job, <method>.confirm.json for a confirmation, and
    <method>@<label>.json for a lab experiment."""
    suffix = ".confirm" if stage == "confirm" else ""
    tag = f"@{label}" if label and label != BASELINE else ""
    return tuning_dir(workspace) / tier / dataset / f"{method}{tag}{suffix}.json"


def read_summary(tier: str, dataset: str, method: str, stage: str = "tune", *, label: str | None = None,
                 workspace: str | None = None) -> dict[str, Any] | None:
    path = summary_path(tier, dataset, method, stage, label=label, workspace=workspace)
    return json.loads(path.read_text()) if path.exists() else None


def _write_summary(summary: dict[str, Any]) -> Path:
    path = summary_path(summary["tier"], summary["dataset"], summary["method"], summary.get("stage", "tune"), label=summary.get("label"))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, default=str))
    tmp.replace(path)
    return path


def prepare_rerun(tier: str, dataset: str, method: str, stage: str = "tune", *, label: str | None = None) -> bool:
    """Make a finished job run again from scratch (after a promotion, or a changed lab experiment): its summary moves
    to archive/ and a stub with the next attempt number takes its place, so the job starts a fresh study and stays
    resumable. A confirmation's summary is only archived: the queue confirms the method again if it is still among
    the dataset's top methods. Returns False when there was no finished job to re-run."""
    path = summary_path(tier, dataset, method, stage, label=label)
    old = json.loads(path.read_text()) if path.exists() else None
    if not old or old.get("status") in (None, "pending", "running"):
        return False
    archive = path.parent / "archive"
    archive.mkdir(exist_ok=True)
    path.replace(archive / f"{path.stem}.{time.strftime('%Y%m%dT%H%M%S')}.json")
    if stage == "confirm":
        return True
    stub = {"tier": tier, "dataset": dataset, "method": method, "label": label, "status": "pending",
            "attempt": int(old.get("attempt", 0)) + 1, "rerun_of": old.get("ended")}
    _write_summary(stub)
    return True


def _storage():
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    tuning_dir().mkdir(parents=True, exist_ok=True)
    return JournalStorage(JournalFileBackend(str(tuning_dir() / "journal.log")))


def _finished_metrics(dataset: str, method: str, cfg: dict[str, Any], split_dir: Path) -> dict[str, float] | None:
    """The metrics of an identical run that already finished (a resumed job, or a repeated configuration)."""
    from recbench.data import TrainView

    run_hash = run_hash_for(cfg, method, TrainView(split_dir).split_hash)
    mlflow = _mlflow()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    frame = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.config_hash = '{run_hash}' and tags.status = 'finished'",
        max_results=1,
    )
    if frame.empty:
        return None
    row = frame.iloc[0]
    return {c[len("metrics."):]: float(row[c]) for c in frame.columns if c.startswith("metrics.") and row[c] == row[c]}  # skip NaN


def _outcome_metrics(outcome: dict[str, Any], dataset: str, method: str, cfg: dict[str, Any], split_dir: Path) -> dict[str, float] | None:
    if outcome.get("status") == "finished":
        return outcome.get("metrics") or {}
    if outcome.get("status") == "skipped_existing":
        return _finished_metrics(dataset, method, cfg, split_dir)
    return None


def _outcome_value(outcome: dict[str, Any], dataset: str, method: str, cfg: dict[str, Any], split_dir: Path) -> float | None:
    return (_outcome_metrics(outcome, dataset, method, cfg, split_dir) or {}).get(METRIC)


def _base_cfg(resolved: dict[str, Any], method: str, tuned_keys: set[str]) -> dict[str, Any]:
    """The resolved config without per-method overrides of tuned keys (they would silently win otherwise)."""
    cfg = dict(resolved)
    own = dict((resolved.get("method_params") or {}).get(method) or {})
    cfg["method_params"] = {**(resolved.get("method_params") or {}), method: {k: v for k, v in own.items() if k not in tuned_keys}}
    return cfg


RERANKERS = ("lgbm_rerank", "dcnv2_rerank")
GENERATORS = {"rerank_ease": "ease", "rerank_itemknn": "itemknn"}  # a re-ranker's setting -> the generator it configures


def size_scale(dataset: str, small_tier: str, big_tier: str) -> float:
    """How many times more users `big_tier`'s validation fold has than `small_tier`'s: the factor by which a setting
    that balances the amount of data (EASE's λ) grows from one to the other (1.0 when a fold is missing)."""
    from recbench.data import TrainView

    root = data_root() / "splits" / dataset
    small, big = root / f"{small_tier}-val", root / f"{big_tier}-val"
    if not ((small / "meta.json").exists() and (big / "meta.json").exists()):
        return 1.0
    return TrainView(big).n_users / max(TrainView(small).n_users, 1)


def scaled_for_size(params: dict[str, Any], space: MethodSpace | None, scale: float) -> dict[str, Any]:
    """`params` with the space's size-sensitive setting (`confirm: {param, scale_with: users}`) multiplied by
    `scale`: the centre value a confirmation would start from."""
    spec = (space.confirm if space else {}) or {}
    name = spec.get("param")
    if spec.get("scale_with") != "users" or name not in params or not isinstance(params[name], (int, float)):
        return params
    return {**params, name: float(params[name]) * scale}


def generator_settings(tier: str, dataset: str, method: str, space: MethodSpace | None = None) -> dict[str, Any]:
    """The settings a re-ranker gives its candidate generator `method` on a split of `tier`: confirmed on that tier,
    else tuned on it, else tuned on the quick tier ({} when not tuned yet). Settings from the smaller quick tier get
    their size-sensitive setting scaled up as a confirmation would (given the generator's `space`). The generator's
    training window is left out: the re-ranker decides which data it trains on."""
    base = tier[: -len("-val")] if tier.endswith("-val") else tier
    for t, stage in dict.fromkeys(((base, "confirm"), (base, "tune"), ("quick", "tune"))):
        summary = read_summary(t, dataset, method, stage)
        if summary and summary.get("best_params"):
            params = {k: v for k, v in summary["best_params"].items() if k != "train_window_days"}
            return scaled_for_size(params, space, size_scale(dataset, t, base)) if t != base else params
    return {}


def pin_generators(cfg: dict[str, Any], method: str, tier: str, dataset: str,
                   spaces: dict[str, MethodSpace] | None = None) -> dict[str, Any]:
    """Write a re-ranker's generator settings into its config (settings already there win). They are then fixed for
    the whole job, whatever finishes meanwhile, and part of every run's identity: when EASE or ItemKNN is tuned
    again, the re-ranker runs again instead of reusing results built on the old candidates. `spaces` (all methods'
    search spaces) lets quick-tier settings be scaled to a bigger tier. Returns the pinned settings ({} for other
    methods)."""
    if method not in RERANKERS:
        return {}
    for key, generator in GENERATORS.items():
        cfg[key] = {**generator_settings(tier, dataset, generator, (spaces or {}).get(generator)), **(cfg.get(key) or {})}
    return {key: cfg[key] for key in GENERATORS}


def run_job(
    dataset: str,
    method: str,
    resolved: dict[str, Any],
    settings: JobSettings,
    space: MethodSpace | None,
    *,
    child_env: dict[str, str] | None = None,
    isolate: bool = True,
    retry: bool = False,
    label: str | None = None,
    fingerprint: str | None = None,
) -> dict[str, Any]:
    """Tune `method` on `dataset` and evaluate the best configuration once. Returns the job summary.

    With retry=True, a job that failed starts a new attempt: a fresh study, so trials that failed (for example
    before a bug fix) do not use up its budget. An interrupted attempt resumes where it stopped.
    `label` names a lab experiment (its own summary and study); `fingerprint` identifies the experiment's definition
    and code, so a changed experiment never continues the study of the old one."""
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
    if label and label != BASELINE:
        summary["label"] = label
    if fingerprint:
        summary["fingerprint"] = fingerprint
    missing = [str(p) for p in (val_dir, test_dir) if not (p / "meta.json").exists()]
    if missing:
        summary.update(status="missing_split", reason=f"prepare these splits first: {missing}", ended=time.time())
        _write_summary(summary)
        return summary

    previous = read_summary(settings.tier, dataset, method, label=label) or {}
    attempt = int(previous.get("attempt", 0)) + (1 if retry and previous.get("status") == "failed" else 0)
    summary.update(attempt=attempt, status="running")
    _write_summary(summary)  # an interrupted job resumes this attempt
    space = space or MethodSpace(method)
    n_trials = space.trials if space.trials is not None else (settings.trials if space.params else 1)
    summary["trials_budget"] = n_trials
    base = _base_cfg(resolved, method, set(space.params) | set(space.fixed))
    base.update({"tuning": "tuned", "export_bundles": False, "resume": True})
    if child_env:
        base["child_env"] = dict(child_env)
    generators = pin_generators(base, method, settings.tier, dataset)
    if generators:
        summary["generators"] = generators
    summary["space"] = {"params": space.params, "fixed": space.fixed}
    summary["impl_version"] = method_version(method, base)
    name = method + (f"@{label}" if label and label != BASELINE else "")
    study = optuna.create_study(
        study_name=f"{settings.tier}/{dataset}/{name}/v{settings.space_version}/s{settings.seed}" + (f"/a{attempt}" if attempt else "")
        + (f"/f{fingerprint}" if fingerprint else ""),
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
        # Training stops (keeping its best epoch) once this trial has used its fair share of the time left, so one
        # slow setting cannot eat the budget of the others. The hard timeout stays at everything that is left.
        fair_share = remaining / (n_trials - len(done) + settings.final_factor)
        cfg = {**base, **params, **(settings.force or {}), "stage": "search", "trial": trial.number, "max_eval_users": settings.search_users,
               "timeout_minutes": trial_budget / 60, "fit_deadline": time.time() + 0.8 * min(fair_share, trial_budget)}
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
        if failed and all(t.user_attrs.get("status") == "unsupported" for t in failed):
            status = "unsupported"  # e.g. an optional package is missing, or the data cannot train this method
        else:
            status = "over_budget" if (stop_reason == "time" or timeouts) else "failed"
        summary.update(status=status, reason=(failed[-1].user_attrs.get("reason") if failed else "no trial fitted in the time cap"),
                       ended=time.time())
        _write_summary(summary)
        return summary

    best = study.best_trial
    best_params = {**space.fixed, **best.params}
    summary.update(best_trial=best.number, best_params=best_params, best_val=best.value, best_epoch=best.user_attrs.get("best_epoch"))
    final_cfg = {**base, **best_params, **(settings.force or {}), "stage": "final"}
    if best.user_attrs.get("best_epoch"):
        final_cfg["epochs"] = int(best.user_attrs["best_epoch"])
    seeds = [int(final_cfg.get("seed", 42))]
    if settings.final_seeds and not ensure_loaded().methods[method].spec.deterministic:
        seeds = [int(s) for s in settings.final_seeds]  # the same configuration per seed: training noise, not selection
    from recbench.data import TrainView

    split_hash = TrainView(test_dir).split_hash
    t0 = time.time()
    finals = []
    for i, seed in enumerate(seeds):
        remaining = deadline - time.time()
        if i and remaining < 60:
            finals.append({"seed": seed, "status": "skipped", "reason": "no time left in the job's cap"})
            continue
        cfg = {**final_cfg, "seed": seed, "timeout_minutes": max(remaining, 60) / 60}
        outcome = run_pair(test_dir, method, cfg, isolate=isolate)
        metrics = _outcome_metrics(outcome, dataset, method, cfg, test_dir) or {}
        finals.append({"seed": seed, "status": outcome.get("status"), "value": metrics.get(METRIC),
                       "config_hash": run_hash_for(cfg, method, split_hash), "metrics": metrics,
                       **({"reason": outcome["reason"]} if outcome.get("reason") else {})})
    summary["final_seconds"] = time.time() - t0
    summary["final_status"] = finals[0]["status"]
    summary["final_runs"] = [{k: f[k] for k in ("seed", "status", "value", "config_hash") if k in f} for f in finals]
    good = [f for f in finals if f.get("value") is not None]
    if good:
        keys = set.intersection(*(set(f["metrics"]) for f in good))
        test = {k: statistics.fmean(float(f["metrics"][k]) for f in good) for k in sorted(keys)}
        if len(good) > 1:
            test[f"{METRIC}_seed_sd"] = statistics.pstdev(float(f["value"]) for f in good)
        summary["test"] = test
        summary["status"] = "finished"
    else:
        first = finals[0]
        summary["test"] = first.get("metrics") or None
        summary["status"] = {"timeout": "over_budget", "unsupported": "unsupported"}.get(str(first.get("status")), "failed")
        summary["reason"] = first.get("reason", first.get("status"))
    summary["ended"] = time.time()
    _write_summary(summary)
    return summary


def confirm_values(space: MethodSpace | None, best_params: dict[str, Any], scale: float) -> list[dict[str, Any]]:
    """Configurations to re-check on full data: the quick-tier best, plus the size-sensitive setting moved by
    `factors` (and multiplied by `scale`, the full/quick user ratio, when `scale_with: users`)."""
    spec = (space.confirm if space else {}) or {}
    name = spec.get("param")
    if not name or name not in best_params or not isinstance(best_params[name], (int, float)):
        return [dict(best_params)]
    base = float(best_params[name]) * (scale if spec.get("scale_with") == "users" else 1.0)
    return [{**best_params, name: base * float(f)} for f in spec.get("factors", [0.5, 1.0, 2.0])]


def run_confirm(
    dataset: str,
    method: str,
    resolved: dict[str, Any],
    settings: JobSettings,
    space: MethodSpace | None,
    *,
    tier: str = "full",
    seeds: list[int] | None = None,
    child_env: dict[str, str] | None = None,
    isolate: bool = True,
    spaces: dict[str, MethodSpace] | None = None,
) -> dict[str, Any]:
    """Re-check a quick-tier winner on full data: a few values of its size-sensitive setting on the full
    validation fold, then the final test run (one per seed for methods whose training is random). `spaces` (all
    methods' search spaces) scales a re-ranker's quick-tier generator settings to the full data."""
    from recbench.data import TrainView

    began = time.time()
    deadline = began + settings.cap_minutes * 60
    quick = read_summary(settings.tier, dataset, method)
    summary: dict[str, Any] = {"tier": tier, "dataset": dataset, "method": method, "stage": "confirm", "started": began,
                               "from_tier": settings.tier, "cap_minutes": settings.cap_minutes}
    root = data_root()
    val_dir, test_dir = root / "splits" / dataset / f"{tier}-val", root / "splits" / dataset / tier
    if not quick or quick.get("status") != "finished":
        summary.update(status="skipped", reason=f"no finished {settings.tier} job to confirm", ended=time.time())
        _write_summary(summary)
        return summary
    missing = [str(p) for p in (val_dir, test_dir) if not (p / "meta.json").exists()]
    if missing:
        summary.update(status="missing_split", reason=f"prepare these splits first: {missing}", ended=time.time())
        _write_summary(summary)
        return summary
    summary["status"] = "running"
    _write_summary(summary)  # an interrupted confirmation stays visible and runs again on resume
    quick_val = root / "splits" / dataset / f"{settings.tier}-val"
    scale = TrainView(val_dir).n_users / max(TrainView(quick_val).n_users, 1)
    base = _base_cfg(resolved, method, set(quick["best_params"]))
    base.update({"tuning": "tuned", "export_bundles": False, "resume": True, "max_eval_users": settings.search_users})
    if child_env:
        base["child_env"] = dict(child_env)
    generators = pin_generators(base, method, tier, dataset, spaces)
    if generators:
        summary["generators"] = generators
    checks = []
    for params in confirm_values(space, quick["best_params"], scale):
        remaining = deadline - time.time()
        if remaining < 120:
            break
        cfg = {**base, **params, **(settings.force or {}), "stage": "confirm", "timeout_minutes": remaining / 60 * 0.5,
               "fit_deadline": time.time() + remaining * 0.4}
        t0 = time.time()
        outcome = run_pair(val_dir, method, cfg, isolate=isolate)
        checks.append({"params": params, "value": _outcome_value(outcome, dataset, method, cfg, val_dir), "status": outcome.get("status"),
                       "seconds": time.time() - t0, "best_epoch": (outcome.get("fit") or {}).get("best_epoch")})
    good = [c for c in checks if c["value"] is not None]
    summary["checks"] = checks
    if not good:
        if checks and all(c["status"] == "timeout" for c in checks):
            status = "over_budget"
        elif checks and all(c["status"] == "unsupported" for c in checks):
            status = "unsupported"
        else:
            status = "failed"
        summary.update(status=status, reason="no confirmation run finished", ended=time.time())
        _write_summary(summary)
        return summary
    best = max(good, key=lambda c: c["value"])
    summary.update(best_params=best["params"], best_val=best["value"], best_epoch=best["best_epoch"])
    spec = ensure_loaded().methods[method].spec
    run_seeds = [int(resolved.get("seed", 42))] if spec.deterministic else list(seeds or [int(resolved.get("seed", 42))])
    finals = []
    for i, seed in enumerate(run_seeds):
        remaining = deadline - time.time()
        if remaining < 60:
            break
        # The first final run also writes the serving bundle (data/bundles/<dataset>/<tier>/<method>), so the API can
        # serve every confirmed method. A run that finished in an earlier session is not repeated: use
        # `python -m recbench.export --from-confirm` for it.
        cfg = {**base, **best["params"], **(settings.force or {}), "seed": seed, "stage": "final", "max_eval_users": resolved.get("max_eval_users"),
               "timeout_minutes": remaining / 60, "export_bundles": i == 0}
        if best["best_epoch"]:
            cfg["epochs"] = int(best["best_epoch"])
        outcome = run_pair(test_dir, method, cfg, isolate=isolate)
        if outcome.get("bundle"):
            summary["bundle"] = outcome["bundle"]
        finals.append({"seed": seed, "status": outcome.get("status"), "value": _outcome_value(outcome, dataset, method, cfg, test_dir)})
    values = [f["value"] for f in finals if f["value"] is not None]
    summary["finals"] = finals
    summary["test"] = {METRIC: statistics.fmean(values), f"{METRIC}_seed_sd": statistics.pstdev(values) if len(values) > 1 else 0.0} if values else None
    summary["status"] = "finished" if values else ("over_budget" if any(f["status"] == "timeout" for f in finals) else "failed")
    summary["ended"] = time.time()
    _write_summary(summary)
    return summary
