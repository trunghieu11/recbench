"""Lab experiments: named changes to a method's bake-off search space, kept in labs/<nn>-<method>/experiments.yaml.

An experiment starts from the method's entry in configs/tuning/quick.yaml (what the baseline searched) and changes it:

    params:   settings added to the search, or searched differently (same format as configs/tuning/quick.yaml)
    remove:   settings no longer searched (the method's default applies)
    fixed:    settings held at one value (they leave the search)
    trials:   settings to try (default: the baseline's budget, 10; keep it for a fair comparison)
    from_baseline: true holds every setting the baseline searched at that dataset's best value and searches only
              `params`: the experiment's 10 trials then all go to the new idea (an ablation)
    note:     what you tried and why (shown on the scoreboard)
    promoted: true once the change is promoted (docs/handbook/promote.md)

The lab folders' numbers set the course order: labs/01-itemknn is week 1, labs/11-lgbm-rerank week 11.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from recbench.config import load_benchmark_yaml, load_yaml
from recbench.paths import repo_root
from recbench.tuning.spaces import MethodSpace, load_spaces

LAB_CONFIG = Path("configs") / "benchmarks" / "lab.yaml"
LABS_DIR = Path("labs")
BASELINE = "baseline"
REFERENCE = "most_popular"  # the floor every method should beat; in the lab's baseline, but not one of the labs
KEYS = {"params", "remove", "fixed", "trials", "from_baseline", "note", "promoted"}
PARAM_TYPES = {"choice", "log", "float", "int", "logint"}
LABEL = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
# Settings read by the runner rather than by a method's own code (see runner.run_single).
RUNNER_KEYS = {"train_window_days", "train_window_keep_last", "seed", "epochs", "max_epochs", "patience", "early_stopping",
               "device", "threads"}


class ExperimentError(ValueError):
    """An experiment file or entry that cannot be used, with a message that says how to fix it."""


@dataclass
class Experiment:
    method: str
    label: str
    space: MethodSpace
    note: str = ""
    promoted: bool = False
    entry: dict[str, Any] = field(default_factory=dict)

    @property
    def is_baseline(self) -> bool:
        return self.label == BASELINE

    @property
    def from_baseline(self) -> bool:
        return bool(self.entry.get("from_baseline"))

    def space_for(self, baseline_best: dict[str, Any] | None) -> MethodSpace:
        """The space actually searched on one dataset. With from_baseline, every setting the baseline chose is held at
        its value there (except those this experiment searches or removes), so only `params` are searched."""
        if not self.from_baseline:
            return self.space
        if not baseline_best:
            raise ExperimentError(f"{self.method}:{self.label} uses from_baseline, but this dataset has no finished baseline")
        own = set(self.entry.get("params") or {})
        removed = set(self.entry.get("remove") or [])
        held = {k: v for k, v in baseline_best.items() if k not in own and k not in removed}
        return MethodSpace(method=self.method, params={k: v for k, v in self.space.params.items() if k in own},
                           fixed={**held, **self.space.fixed}, trials=self.space.trials, confirm=dict(self.space.confirm))


def lab_benchmark(root: Path | None = None) -> dict[str, Any]:
    """configs/benchmarks/lab.yaml, merged with the file it extends (no side effects)."""
    root = root or repo_root()
    return load_benchmark_yaml(root / LAB_CONFIG, root)


def lab_datasets(root: Path | None = None) -> list[str]:
    return list(lab_benchmark(root).get("datasets") or [])


def lab_folders(root: Path | None = None) -> dict[str, Path]:
    """{method: labs/<nn>-<method>} in course order (the folder numbers)."""
    root = root or repo_root()
    found = {}
    for folder in sorted((root / LABS_DIR).glob("[0-9][0-9]-*")):
        if folder.is_dir():
            found[folder.name[3:].replace("-", "_")] = folder
    return found


def lab_methods(root: Path | None = None) -> list[str]:
    """The methods with a lab, in course order."""
    return list(lab_folders(root))


def experiments_file(method: str, root: Path | None = None) -> Path:
    folders = lab_folders(root)
    if method not in folders:
        raise ExperimentError(f"{method} has no lab folder under labs/ (the labs are: {', '.join(folders) or 'none'})")
    return folders[method] / "experiments.yaml"


def load_experiments(method: str, root: Path | None = None) -> dict[str, dict[str, Any]]:
    """{label: entry} from the method's experiments.yaml, checked."""
    path = experiments_file(method, root)
    raw = load_yaml(path) if path.exists() else {}
    entries = raw.get("experiments") or {}
    if not isinstance(entries, dict):
        raise ExperimentError(f"{path}: `experiments` must be a mapping of label -> settings")
    out = {}
    for label, entry in entries.items():
        label, entry = str(label), entry or {}
        if not LABEL.match(label):
            raise ExperimentError(f"{path}: label {label!r} must be lower-case letters, digits and dashes (for example `no-decay`)")
        if not isinstance(entry, dict) or set(entry) - KEYS:
            raise ExperimentError(f"{path}: {label}: unknown keys {sorted(set(entry) - KEYS)}; allowed: {sorted(KEYS)}")
        if label == BASELINE and set(entry) - {"note"}:
            raise ExperimentError(f"{path}: `baseline` must stay unchanged (it is the reference); copy it under a new label")
        if entry.get("from_baseline") and not entry.get("params"):
            raise ExperimentError(f"{path}: {label}: from_baseline searches only `params`, and there are none")
        out[label] = entry
    out.setdefault(BASELINE, {})
    return out


def bake_off_space(method: str, root: Path | None = None) -> MethodSpace:
    """The method's search space in the bake-off (what the baseline searched)."""
    root = root or repo_root()
    tuning = lab_benchmark(root).get("tuning") or {}
    spaces, _ = load_spaces(root / tuning.get("spaces", "configs/tuning/quick.yaml"))
    return spaces.get(method) or MethodSpace(method)


def build_space(method: str, entry: dict[str, Any], base: MethodSpace) -> MethodSpace:
    """The baseline's space changed by one experiment entry."""
    params = dict(base.params)
    for name in entry.get("remove") or []:
        if name not in params:
            raise ExperimentError(f"{method}: cannot remove {name!r}: the baseline does not search it (it searches {sorted(params)})")
        del params[name]
    for name, spec in (entry.get("params") or {}).items():
        if not isinstance(spec, dict) or spec.get("type", "choice") not in PARAM_TYPES:
            raise ExperimentError(f"{method}: {name}: write it as in configs/tuning/quick.yaml, for example "
                                  f"{{type: choice, values: [...]}} (types: {sorted(PARAM_TYPES)})")
        if spec.get("type", "choice") == "choice" and not spec.get("values"):
            raise ExperimentError(f"{method}: {name}: a choice needs `values`")
        params[name] = dict(spec)
    fixed = {**base.fixed, **(entry.get("fixed") or {})}
    for name in fixed:
        params.pop(name, None)  # a fixed setting is not searched (a searched value would win over it)
    trials = entry.get("trials", base.trials)
    if trials is not None and (not isinstance(trials, int) or trials < 1):
        raise ExperimentError(f"{method}: trials must be a whole number of at least 1")
    return MethodSpace(method=method, params=params, fixed=fixed, trials=trials, confirm=dict(base.confirm))


def get(method: str, label: str = BASELINE, root: Path | None = None) -> Experiment:
    entries = load_experiments(method, root)
    if label not in entries:
        raise ExperimentError(f"{method} has no experiment {label!r} in {experiments_file(method, root)} "
                              f"(it has: {', '.join(entries)})")
    entry = entries[label]
    space = build_space(method, entry, bake_off_space(method, root))
    return Experiment(method, label, space, str(entry.get("note") or ""), bool(entry.get("promoted")), entry)


def fingerprint(experiment: Experiment, code_version: str) -> str:
    """Identifies an experiment's definition and the code it runs (`code_version`: runner.method_version)."""
    space = experiment.space
    payload = {"params": space.params, "fixed": space.fixed, "trials": space.trials, "code": code_version,
               "from_baseline": experiment.from_baseline}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]


def unread_settings(method: str, names: set[str] | list[str]) -> list[str]:
    """Settings that no code of the method reads, most likely because a variant is not written yet (or a typo).

    A method reads its settings as cfg.get("name"), so a name that appears nowhere in its code has no effect."""
    import sys

    from recbench.registry import ensure_loaded

    module = sys.modules[ensure_loaded().methods[method].__module__]
    path = Path(module.__file__)
    files = [path] + [path.with_name(f"{name}.py") for name in re.findall(r"recbench\.methods\.(\w+)", path.read_text())]
    files.append(path.parents[1] / "data.py")
    source = "\n".join(f.read_text() for f in files if f.exists())
    return sorted(n for n in names if n not in RUNNER_KEYS and f'"{n}"' not in source and f"'{n}'" not in source)
