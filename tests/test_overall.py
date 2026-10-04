"""The overall comparison: ranks and ties across datasets, mean rank on the common set, relative score, critical
difference, Pareto front, and a well-formed chart."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pandas as pd
import pytest

from recbench.report.overall import critical_difference, pareto_front, render_matrix, results_matrix, svg_scatter, cost_table


def _run(method: str, dataset: str, ndcg: float, half_width: float = 0.001, train: float = 10.0) -> dict:
    return {"tags.method": method, "tags.dataset": dataset, "tags.ranked": "true", "metrics.ndcg_at_10": ndcg,
            "metrics.ndcg_at_10_ci_low": ndcg - half_width, "metrics.ndcg_at_10_ci_high": ndcg + half_width,
            "metrics.train_seconds": train, "metrics.peak_gpu_mb": 0.0, "metrics.peak_rss_mb": 100.0,
            "metrics.score_seconds_per_1k_users": 0.1}


@pytest.fixture()
def frame() -> pd.DataFrame:
    # The worked example of docs/results/overall-comparison.md, plus D, which ran on one dataset only.
    return pd.DataFrame([
        _run("A", "movielens-25m", 0.20), _run("B", "movielens-25m", 0.15), _run("C", "movielens-25m", 0.05),
        _run("A", "retailrocket", 0.010), _run("B", "retailrocket", 0.012, half_width=0.003), _run("C", "retailrocket", 0.004),
        _run("D", "movielens-25m", 0.199, half_width=0.01),
    ])


def test_matrix_ranks_ties_and_summaries(frame):
    table, info = results_matrix(frame)
    assert info["datasets"] == ["movielens-25m", "retailrocket"]  # run order, not alphabetical
    assert sorted(info["common"]) == ["A", "B", "C"]  # D missed a dataset
    assert table.loc["A", "movielens-25m:rank"] == 1 and table.loc["B", "retailrocket:rank"] == 1
    assert table.loc["D", "movielens-25m:tied"] and table.loc["A", "retailrocket:tied"]  # overlapping intervals
    assert table.loc["A", "mean_rank"] == pytest.approx(1.5) and table.loc["B", "mean_rank"] == pytest.approx(1.5)
    assert table.loc["C", "mean_rank"] == pytest.approx(3.0) and pd.isna(table.loc["D", "mean_rank"])
    assert table.loc["A", "relative"] == pytest.approx((1.0 + 0.010 / 0.012) / 2)
    assert table.loc["A", "wins"] == 1 and table.loc["A", "ties"] == 2  # a win counts as a tie with the best
    assert table.loc["C", "ties"] == 0  # a missing dataset never counts as a tie
    text = render_matrix(table, info)
    assert "**1st**" in text and "Critical difference" in text and "—" in text  # D's missing dataset


def test_critical_difference_matches_demsar():
    assert critical_difference(3, 5) == pytest.approx(1.48, abs=0.01)  # q = 2.343 (Demšar 2006, table 5)
    assert critical_difference(1, 5) is None and critical_difference(3, 1) is None


def test_pareto_front_keeps_only_undominated_methods():
    points = {"fast_ok": (0.7, 1.0), "slow_best": (0.9, 100.0), "slow_worse": (0.6, 50.0), "same_but_slower": (0.7, 2.0)}
    assert pareto_front(points) == {"fast_ok", "slow_best"}


def test_chart_is_well_formed_svg(frame):
    table, _ = results_matrix(frame)
    cost = cost_table(frame, table, {"A": "GPU"}, {})
    svg = svg_scatter(cost, "a & b")
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg") and svg.count("<circle") >= len(cost)
