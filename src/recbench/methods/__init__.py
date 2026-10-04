"""Importing this package registers every method (see recbench.registry).

A child process that runs a single method gets RECBENCH_METHOD_MODULES (for example "linear") from
runner.run_pair, and imports only that module. Methods that do not use PyTorch then never load it. This
matters on macOS, where PyTorch's OpenMP runtime and the one loaded by SuiteSparse (SANSA) abort or crash
the process when both are in it.
"""

from __future__ import annotations

import importlib
import os

MODULES = (
    "baselines", "content", "dcnv2", "graph", "graph_filters", "gru4rec", "hstu", "implicit_mf", "linear", "mf_losses",
    "neighbourhood", "recbole_models", "recombee", "rerank", "sasrec", "text_knn", "tiger", "ultragcn", "vae",
)

for _name in [m for m in os.environ.get("RECBENCH_METHOD_MODULES", "").split(",") if m] or MODULES:
    importlib.import_module(f"recbench.methods.{_name}")

__all__ = list(MODULES)
