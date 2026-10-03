"""Importing this package registers every method (see recbench.registry)."""

from recbench.methods import (
    baselines,
    content,
    dcnv2,
    graph,
    graph_filters,
    hstu,
    implicit_mf,
    linear,
    mf_losses,
    neighbourhood,
    recbole_models,
    recombee,
    sasrec,
    tiger,
    ultragcn,
)

__all__ = ["baselines", "content", "dcnv2", "graph", "graph_filters", "hstu", "implicit_mf", "linear", "mf_losses", "neighbourhood",
           "recbole_models", "recombee", "sasrec", "tiger", "ultragcn"]
