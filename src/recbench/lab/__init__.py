"""The improvement lab: improve the light methods yourself and judge each change fairly (docs/labs/index.md).

In a notebook:

    from recbench import lab
    lab.setup()                                     # first: light methods only, the lab workspace, a health check
    data, split = lab.load("movielens-25m")         # the validation fold
    model = lab.fit("itemknn", data, knn_neighbors=200)
    lab.evaluate(model, data, split)                # the same numbers as `python -m recbench.lab once`

In a terminal: python -m recbench.lab --help and python -m recbench.compare --help.
"""

from recbench.lab.api import (
    Evaluation,
    baseline,
    compare,
    config,
    evaluate,
    find_items,
    fit,
    load,
    neighbours,
    once,
    plot_segments,
    plot_sweep,
    results,
    run,
    segments,
    setup,
    sweep,
    trials,
)

__all__ = ["Evaluation", "baseline", "compare", "config", "evaluate", "find_items", "fit", "load", "neighbours", "once",
           "plot_segments", "plot_sweep", "results", "run", "segments", "setup", "sweep", "trials"]
