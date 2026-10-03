"""Importing this package registers every method (see recbench.registry)."""

from recbench.methods import (
    baselines,
    content,
    dcnv2,
    graph,
    hstu,
    implicit_mf,
    linear,
    neighbourhood,
    recbole_models,
    recombee,
    sasrec,
    tiger,
)

__all__ = ["baselines", "content", "dcnv2", "graph", "hstu", "implicit_mf", "linear", "neighbourhood", "recbole_models",
           "recombee", "sasrec", "tiger"]
