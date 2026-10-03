"""Search spaces: which settings each method may tune, and how a trial samples them.

The spaces live in a YAML file (configs/tuning/quick.yaml), so they can be read and changed without
touching code. A parameter is one of:

    {type: choice, values: [...]}         one of the listed values (null allowed)
    {type: log, low: 1e-3, high: 10}      a float, sampled uniformly on a log scale
    {type: float, low: 0.0, high: 0.5}    a float, sampled uniformly
    {type: int, low: 1, high: 4}          an integer
    {type: logint, low: 10, high: 1000}   an integer, sampled on a log scale

`common` parameters (for example train_window_days) are added to every method unless the method sets
`common: false`. A method may also fix settings with `fixed: {...}` and change its trial budget with `trials`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from recbench.config import load_yaml


@dataclass
class MethodSpace:
    method: str
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    fixed: dict[str, Any] = field(default_factory=dict)
    trials: int | None = None
    confirm: dict[str, Any] = field(default_factory=dict)

    def sample(self, trial: Any) -> dict[str, Any]:
        """Draw one configuration through an Optuna trial (suggest_* calls)."""
        values: dict[str, Any] = {}
        for name, spec in self.params.items():
            kind = spec.get("type", "choice")
            if kind == "choice":
                values[name] = trial.suggest_categorical(name, list(spec["values"]))
            elif kind == "log":
                values[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
            elif kind == "float":
                values[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
            elif kind == "int":
                values[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
            elif kind == "logint":
                values[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), log=True)
            else:
                raise ValueError(f"{self.method}.{name}: unknown parameter type {kind}")
        return {**self.fixed, **values}


def load_spaces(path: str | Path) -> tuple[dict[str, MethodSpace], dict[str, Any]]:
    """Read a spaces file. Returns ({method: MethodSpace}, file-level settings such as `version`)."""
    raw = load_yaml(path)
    common = raw.get("common") or {}
    spaces: dict[str, MethodSpace] = {}
    for method, entry in (raw.get("methods") or {}).items():
        entry = entry or {}
        params = dict(common) if entry.get("common", True) else {}
        params.update(entry.get("params") or {})
        spaces[method] = MethodSpace(
            method=method,
            params=params,
            fixed=dict(entry.get("fixed") or {}),
            trials=entry.get("trials"),
            confirm=dict(entry.get("confirm") or {}),
        )
    settings = {k: v for k, v in raw.items() if k not in {"common", "methods"}}
    return spaces, settings
