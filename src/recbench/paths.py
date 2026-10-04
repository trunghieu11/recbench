"""Where recbench reads and writes: the repository, and the results of runs.

Runs normally write to runs/ and reports/: the bake-off's results. A *workspace* keeps another set of results apart.
With the workspace "lab" (a benchmark file says `workspace: lab`), everything a run writes goes to runs/lab/ and
reports/lab/ instead: the MLflow store, tuning summaries, queue state and reports. Lab experiments then never mix with
the bake-off, and fetching the bake-off's results never overwrites them.

The workspace travels to child processes in the RECBENCH_WORKSPACE environment variable. This module imports nothing
heavy, so the serving image can use it.
"""

from __future__ import annotations

import os
from pathlib import Path

WORKSPACE_ENV = "RECBENCH_WORKSPACE"


def repo_root() -> Path:
    return Path(os.environ.get("RECBENCH_ROOT", Path.cwd())).resolve()


def workspace() -> str:
    """The active workspace: "" for the bake-off, or a name such as "lab"."""
    return os.environ.get(WORKSPACE_ENV, "").strip().strip("/")


def use_workspace(name: str | None) -> None:
    """Make this process, and the child processes it starts, use workspace `name` (None or "": the bake-off)."""
    if name:
        os.environ[WORKSPACE_ENV] = str(name).strip().strip("/")
    else:
        os.environ.pop(WORKSPACE_ENV, None)


def runs_dir(name: str | None = None) -> Path:
    """runs/ for the bake-off, runs/<workspace>/ otherwise. `name` reads another workspace ("" is the bake-off)."""
    name = workspace() if name is None else name
    return repo_root() / "runs" / name if name else repo_root() / "runs"


def reports_dir(name: str | None = None) -> Path:
    """reports/ for the bake-off, reports/<workspace>/ otherwise."""
    name = workspace() if name is None else name
    return repo_root() / "reports" / name if name else repo_root() / "reports"


def tracking_uri(name: str | None = None) -> str:
    """The MLflow store. In a workspace it is always runs/<workspace>/mlflow, so a MLFLOW_TRACKING_URI left in the
    shell cannot send lab runs to the bake-off's store. For the bake-off: $MLFLOW_TRACKING_URI, else runs/mlflow."""
    name = workspace() if name is None else name
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri and not name:
        if uri.startswith("file:") and not uri.startswith("file:///"):
            return (repo_root() / uri[len("file:"):]).resolve().as_uri()
        return uri
    return (runs_dir(name) / "mlflow").resolve().as_uri()
